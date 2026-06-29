"""
5-phase finance pretraining for BushidoMythos.

Phase 1 — General fluency:
    WikiText-103 (~135M tokens, 20 000 steps)
Phase 2 — Reasoning:
    open-web-math/open-web-math  (60%: mathematical web text, quantitative reasoning)
    microsoft/orca-math-word-problems-200k (math word-problem solutions)
    databricks/databricks-dolly-15k (instruction-following prose reasoning)
    Combined: ~80K OpenWebMath rows + ~47K Orca rows + ~15K Dolly rows, 8 000 steps
Phase 3 — Finance domain + instruction tuning:
    ashraq/financial-news-articles (~306K articles, plain text)
    gbharti/finance-alpaca (~21K instruction examples, formatted as plain text)
    Combined: 8 000 steps
Phase 4 — Trading methodology SFT:
    FinGPT/fingpt-forecaster-dow30-202305-202405 (~1.2K stock-movement prediction examples)
    FinGPT/fingpt-sentiment-train (~76K sentiment-analysis examples)
    Combined: ~78K pairs, 3 000 steps
Phase 5 — Trading discipline / risk-management QA  ← FINAL calibration phase:
    FinGPT/fingpt-fiqa_qa (~17K financial QA examples, 3 000 steps)
    Last-mile SFT anchors the model's response style on risk acknowledgement
    and uncertainty disclosure before deployment.

Usage:
    # Full run (all 5 phases):
    python training/finance_pretrain.py --phase 0

    # Resume an interrupted run:
    python training/finance_pretrain.py --auto_resume

    # Phase 3 only (instruction tuning from phase2_final.pt):
    python training/finance_pretrain.py --phase 3 --resume checkpoints/finance_a100_v2/phase2_final.pt

    # Phase 4 only (trading methodology from phase3_final.pt):
    python training/finance_pretrain.py --phase 4 --resume checkpoints/finance_a100_v2/phase3_final.pt

    # Phase 5 only (risk-management QA from phase4_final.pt):
    python training/finance_pretrain.py --phase 5 --resume checkpoints/finance_a100_v2/phase4_final.pt

    # Quick smoke test:
    python training/finance_pretrain.py --phase 1 --phase1_steps 100 --log_every 10
"""

from __future__ import annotations  # アノテーション遅延評価(int | None 等を Py3.9 でも許可)

import argparse
import datetime
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW


class _Tee:
    """Duplicate stdout writes to a log file with timestamps."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data: str) -> int:
        if data:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for line in data.splitlines(keepends=True):
                timestamped = f"[{ts}] {line}" if line.strip() else line
                self._file.write(timestamped)
            self._stdout.write(data)
        return len(data)

    def flush(self) -> None:
        self._file.flush()
        self._stdout.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos


# ──────────────────────────────────────────────────────────────
# VRAM diagnostics
# ──────────────────────────────────────────────────────────────

def _vram_str(device: torch.device, reset_peak: bool = True) -> str:
    """Return a formatted VRAM status string (empty string on non-CUDA devices).

    Fields:
      alloc    -- bytes currently held by live tensors
      reserved -- bytes reserved from CUDA (alloc + allocator cache)
      peak     -- max alloc since last reset (resets after each call when reset_peak=True)
      frag     -- (reserved - alloc) / reserved  →  high = fragmentation or cached blocks

    Diagnosis guide:
      alloc grows steadily         → activations not freed; check grad-checkpoint
      frag > 40% and rising        → allocator fragmentation; try PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      peak >> steady alloc         → transient spike (e.g. attention matrix); consider SDPA or seq-len reduction
      reserved near GPU total      → OOM imminent; reduce batch or enable grad-checkpoint
    """
    if device.type != "cuda":
        return ""
    MB = 1024 ** 2
    alloc    = torch.cuda.memory_allocated(device) / MB
    reserved = torch.cuda.memory_reserved(device) / MB
    peak     = torch.cuda.max_memory_allocated(device) / MB
    frag     = (reserved - alloc) / reserved * 100 if reserved > 0 else 0.0
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    return (
        f"alloc={alloc:.0f}MB  reserved={reserved:.0f}MB"
        f"  peak={peak:.0f}MB  frag={frag:.0f}%"
    )


# ──────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────

def get_device_and_dtype(dtype_arg: str) -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if dtype_arg == "auto":
            # bfloat16 requires Ampere (SM 8.0+). T4 is Turing (SM 7.5) — use float16.
            cc_major = torch.cuda.get_device_properties(0).major
            dtype = torch.bfloat16 if cc_major >= 8 else torch.float16
        else:
            dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_arg]
        return device, dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


# ──────────────────────────────────────────────────────────────
# Tokenizer helper
# ──────────────────────────────────────────────────────────────

def _get_gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


# ──────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────

# データサンプリング(batch 順序)専用の seed。train() が args.seed で上書きする。
# モデル構築が消費する torch global RNG とは分離した torch.Generator に使うことで、
# 構造が違う MLA/GQA を別プロセスで学習しても batch 順序が一致する。
_DATA_SAMPLE_SEED = 42


class TextDataset:
    """
    Generic flat-token dataset built from a list of text rows.
    Tokenises row-by-row, concatenates, then serves fixed-length chunks.
    Disk-caches the token tensor so the second run is instant.
    """

    def __init__(
        self,
        rows: list[str],
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        cache_path: Path,
        sample_seed: int | None = None,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        # batch 順序用の専用 Generator(モデル RNG 消費と分離 → MLA/GQA で順序一致)
        self._gen = torch.Generator()
        self._gen.manual_seed(_DATA_SAMPLE_SEED if sample_seed is None else sample_seed)

        if cache_path.exists():
            print(f"  Loading cached tokens from {cache_path}")
            self._ids = torch.load(cache_path, weights_only=True)
        else:
            print(f"  Tokenising {len(rows):,} rows (will cache to {cache_path}) …")
            tok = _get_gpt2_tokenizer()
            all_ids: list[int] = []
            for i, row in enumerate(rows):
                row = row.strip()
                if not row:
                    continue
                toks = tok.encode(row, add_special_tokens=False)
                all_ids.extend(min(t, vocab_size - 1) for t in toks)
                if (i + 1) % 50_000 == 0:
                    print(f"    … {i+1:,}/{len(rows):,} rows ({len(all_ids):,} tokens)")
            self._ids = torch.tensor(all_ids, dtype=torch.long)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._ids, cache_path)
            print(f"  Cached {len(self._ids):,} tokens → {cache_path}")

    def __len__(self) -> int:
        return max(1, (len(self._ids) - self.seq_len - 1) // (self.batch_size * self.seq_len))

    def __iter__(self):
        ids = self._ids
        n = len(ids) - self.seq_len - 1
        if n <= 0:
            raise ValueError(
                f"Token count ({len(ids):,}) is too small for seq_len={self.seq_len}. "
                "Check your cache or reduce --seq_len."
            )
        starts = torch.randperm(n, generator=self._gen)[: len(self) * self.batch_size]
        for i in range(0, len(starts) - self.batch_size + 1, self.batch_size):
            batch_starts = starts[i : i + self.batch_size]
            x = torch.stack([ids[s : s + self.seq_len] for s in batch_starts]).to(self.device)
            y = torch.stack([ids[s + 1 : s + self.seq_len + 1] for s in batch_starts]).to(self.device)
            yield x, y


# Bump this when tokenization logic changes to invalidate old caches.
_CACHE_VERSION = "v1"


def build_wikitext103(vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str) -> TextDataset:
    from datasets import load_dataset
    print("Loading WikiText-103 …")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    rows = ds["text"]
    print(f"  {len(rows):,} rows loaded from WikiText-103.")
    cache_path = Path(cache_dir) / f"wikitext103_gpt2_{vocab_size}_{_CACHE_VERSION}.pt"
    dataset = TextDataset(rows, vocab_size, seq_len, batch_size, device, cache_path)
    del rows  # free list; token tensor is in dataset._ids
    return dataset


def build_financial_news(vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str) -> TextDataset:
    from datasets import load_dataset
    print("Loading ashraq/financial-news-articles …")
    ds = load_dataset("ashraq/financial-news-articles", split="train")

    # Validate expected columns exist
    sample = ds[0]
    available = set(sample.keys())
    required = {"title", "text"}
    if not required & available:
        raise RuntimeError(
            f"financial-news-articles has no 'title' or 'text' column. "
            f"Available columns: {sorted(available)}"
        )

    rows: list[str] = []
    for item in ds:
        title = (item.get("title") or "").strip()
        text  = (item.get("text")  or "").strip()
        if title and text:
            rows.append(f"{title}\n{text}")
        elif text:
            rows.append(text)
        elif title:
            rows.append(title)
    print(f"  {len(rows):,} articles loaded.")
    cache_path = Path(cache_dir) / f"financial_news_gpt2_{vocab_size}_{_CACHE_VERSION}.pt"

    dataset = TextDataset(rows, vocab_size, seq_len, batch_size, device, cache_path)
    del rows
    return dataset


# ──────────────────────────────────────────────────────────────
# Instruction-tuning helpers (Phase 3 & 4)
# ──────────────────────────────────────────────────────────────

_INSTRUCT_EOS = "<|endoftext|>"  # GPT-2 EOS — marks end of each example


def _format_instruct(instruction: str, response: str, context: str = "") -> str:
    """Format one instruction-response pair into the canonical training string."""
    parts = [f"### Instruction:\n{instruction.strip()}"]
    if context.strip():
        parts.append(f"### Input:\n{context.strip()}")
    parts.append(f"### Response:\n{response.strip()}")
    return "\n\n".join(parts) + _INSTRUCT_EOS


def _tokenize_sft(
    tok, instruction: str, response: str, context: str, vocab_size: int
) -> tuple[list[int], list[bool]]:
    """Tokenize one pair; loss_mask=True for response tokens only."""
    if context.strip():
        prompt = (f"### Instruction:\n{instruction.strip()}\n\n"
                  f"### Input:\n{context.strip()}\n\n### Response:\n")
    else:
        prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
    response_text = response.strip() + _INSTRUCT_EOS

    prompt_ids   = [min(t, vocab_size - 1) for t in tok.encode(prompt,         add_special_tokens=False)]
    response_ids = [min(t, vocab_size - 1) for t in tok.encode(response_text,  add_special_tokens=False)]
    ids  = prompt_ids + response_ids
    mask = [False] * len(prompt_ids) + [True] * len(response_ids)
    return ids, mask


class SFTDataset:
    """
    Instruction-tuning dataset with response-only loss masking.
    Yields (x, y, loss_mask) where loss_mask=True only for response tokens.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str, str]],  # (instruction, response, context)
        vocab_size: int,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        cache_path: Path,
        sample_seed: int | None = None,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        # batch 順序用の専用 Generator(モデル RNG 消費と分離 → MLA/GQA で順序一致)
        self._gen = torch.Generator()
        self._gen.manual_seed(_DATA_SAMPLE_SEED if sample_seed is None else sample_seed)

        if cache_path.exists():
            print(f"  Loading cached SFT tokens from {cache_path}")
            saved = torch.load(cache_path, weights_only=True)
            self._ids  = saved["ids"]
            self._mask = saved["mask"]
        else:
            print(f"  Tokenising {len(pairs):,} instruction pairs (will cache to {cache_path}) …")
            tok = _get_gpt2_tokenizer()
            all_ids:  list[int]  = []
            all_mask: list[bool] = []
            for i, (inst, resp, ctx) in enumerate(pairs):
                ids, mask = _tokenize_sft(tok, inst, resp, ctx, vocab_size)
                all_ids.extend(ids)
                all_mask.extend(mask)
                if (i + 1) % 5_000 == 0:
                    print(f"    … {i+1:,}/{len(pairs):,} pairs ({len(all_ids):,} tokens)")
            self._ids  = torch.tensor(all_ids,  dtype=torch.long)
            self._mask = torch.tensor(all_mask, dtype=torch.bool)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"ids": self._ids, "mask": self._mask}, cache_path)
            print(f"  Cached {len(self._ids):,} tokens → {cache_path}")

    def __len__(self) -> int:
        return max(1, (len(self._ids) - self.seq_len - 1) // (self.batch_size * self.seq_len))

    def __iter__(self):
        n = len(self._ids) - self.seq_len - 1
        if n <= 0:
            raise ValueError(f"Token count ({len(self._ids):,}) too small for seq_len={self.seq_len}.")
        starts = torch.randperm(n, generator=self._gen)[: len(self) * self.batch_size]
        for i in range(0, len(starts) - self.batch_size + 1, self.batch_size):
            bs = starts[i : i + self.batch_size]
            x    = torch.stack([self._ids [s : s + self.seq_len    ] for s in bs]).to(self.device)
            y    = torch.stack([self._ids [s + 1 : s + self.seq_len + 1] for s in bs]).to(self.device)
            mask = torch.stack([self._mask[s + 1 : s + self.seq_len + 1] for s in bs]).to(self.device)
            yield x, y, mask


def _extract_sft_pairs(
    ds,
    name: str,
    instruction_cols: list,
    output_cols: list,
    input_cols: list = None,
) -> list:
    """Flexible column extractor for SFT datasets.

    Tries each candidate column name in order, uses the first that exists.
    Skips rows where instruction or output is empty and reports counts.
    Raises KeyError with diagnostics if required columns are absent.
    """
    available = set(ds.column_names)
    inst_col = next((c for c in instruction_cols if c in available), None)
    out_col  = next((c for c in output_cols      if c in available), None)
    in_col   = next((c for c in (input_cols or []) if c in available), None)

    if inst_col is None or out_col is None:
        raise KeyError(
            f"[{name}] Required columns not found.\n"
            f"  Tried instruction: {instruction_cols}\n"
            f"  Tried output:      {output_cols}\n"
            f"  Available:         {sorted(available)}"
        )

    col_info = f"instruction='{inst_col}' output='{out_col}'"
    if in_col:
        col_info += f" input='{in_col}'"
    print(f"  Columns: {col_info}")

    pairs, skipped = [], 0
    for r in ds:
        inst = str(r.get(inst_col) or "").strip()
        resp = str(r.get(out_col)  or "").strip()
        ctx  = str(r.get(in_col)   or "").strip() if in_col else ""
        if not inst or not resp:
            skipped += 1
            continue
        pairs.append((inst, resp, ctx))

    print(f"  Loaded: {len(pairs):,}  Skipped (empty): {skipped:,}")
    return pairs


def build_reasoning_mix(
    vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str,
    openwebmath_rows: int = 80_000,
    orca_ratio: float = 35.0,
    include_dolly: bool = False,
    dolly_rows: int = 15_000,
) -> TextDataset:
    """Phase 2: OpenWebMath + Orca Math [+ Dolly] — quantitative and prose reasoning.

    OpenWebMath: mathematical web text (equations, proofs, derivations).
    Orca Math: chain-of-thought word-problem solutions — step-by-step decomposition.
    Dolly (opt-in, --include_dolly): prose instruction-response pairs for explanation /
        judgment / constraint-handling. License: CC BY-SA 3.0 (attribution + share-alike).
        Enable only when your use-case is compatible with CC BY-SA obligations.

    openwebmath_rows: rows to stream from OpenWebMath (full dataset = 6.3B tokens).
    orca_ratio: Orca rows as % of openwebmath_rows (default 35 → ~47K rows).
    include_dolly: opt-in flag for Dolly (CC BY-SA 3.0). Off by default.
    dolly_rows: cap on Dolly rows when include_dolly=True.
    """
    import random
    from datasets import load_dataset

    random.seed(42)
    rows: list[str] = []

    # ── OpenWebMath (streamed to avoid downloading full 6.3B-token corpus) ──
    print(f"Loading open-web-math/open-web-math (streaming, cap={openwebmath_rows:,}) …")
    ds_owm = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for i, ex in enumerate(ds_owm):
        if i >= openwebmath_rows:
            break
        text = (ex.get("text") or "").strip()
        if len(text) > 100:
            rows.append(text)
    owm_count = len(rows)
    print(f"  OpenWebMath: {owm_count:,} rows")

    # ── Orca Math CoT word problems ──────────────────────────────────────────
    target_orca = int(owm_count * orca_ratio / 100)
    print(f"Loading microsoft/orca-math-word-problems-200k (target {target_orca:,} rows) …")
    ds_orca = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    orca_items = list(ds_orca)
    random.shuffle(orca_items)
    for item in orca_items[:target_orca]:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            rows.append(f"Question: {q}\nSolution: {a}")
    print(f"  Orca Math: {len(rows) - owm_count:,} rows")

    # ── Dolly prose reasoning (opt-in, CC BY-SA 3.0) ────────────────────────
    dolly_count = 0
    if include_dolly:
        before_dolly = len(rows)
        print(f"Loading databricks/databricks-dolly-15k (CC BY-SA 3.0, cap={dolly_rows:,}) …")
        ds_dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        dolly_items = list(ds_dolly)
        random.shuffle(dolly_items)
        for item in dolly_items[:dolly_rows]:
            inst = (item.get("instruction") or "").strip()
            ctx  = (item.get("context")     or "").strip()
            resp = (item.get("response")    or "").strip()
            if not inst or not resp:
                continue
            parts = [f"Instruction: {inst}"]
            if ctx:
                parts.append(f"Context: {ctx}")
            parts.append(f"Response: {resp}")
            rows.append("\n".join(parts))
        dolly_count = len(rows) - before_dolly
        print(f"  Dolly prose reasoning: {dolly_count:,} rows (CC BY-SA 3.0)")
    else:
        print("  Dolly: skipped (use --include_dolly to enable; license: CC BY-SA 3.0)")

    random.shuffle(rows)
    print(f"  Total reasoning mix: {len(rows):,} rows"
          f"  (OWM={owm_count:,} Orca={len(rows)-owm_count-dolly_count:,} Dolly={dolly_count:,})")

    # キャッシュ名にデータ構成バージョンを含める（配合変更時の誤 hit を防止）
    _MIX_VERSION = "rmv1"
    dolly_tag = f"_dolly{dolly_rows}" if include_dolly else "_nodolly"
    cache_path = Path(cache_dir) / (
        f"reasoning_mix_gpt2_{vocab_size}"
        f"_owm{openwebmath_rows}_orca{int(orca_ratio)}{dolly_tag}_{_MIX_VERSION}_{_CACHE_VERSION}.pt"
    )
    dataset = TextDataset(rows, vocab_size, seq_len, batch_size, device, cache_path)
    del rows
    return dataset


def build_finance_domain_mix(
    vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str,
) -> TextDataset:
    """Phase 3: financial-news-articles + finance-alpaca as plain text (domain + instruction exposure).

    Both datasets are combined as plain next-token-prediction text (no loss masking).
    Loss masking is reserved for Phase 4+ SFT where precise response anchoring matters.
    finance-alpaca rows are formatted with the canonical ### Instruction / ### Response template
    so the model learns the format naturally before Phase 4 SFT enforces it with masking.
    """
    import random
    from datasets import load_dataset

    rows: list[str] = []

    # ── Financial news (plain text) ─────────────────────────────────────────
    print("Loading ashraq/financial-news-articles …")
    ds_news = load_dataset("ashraq/financial-news-articles", split="train")
    for item in ds_news:
        title = (item.get("title") or "").strip()
        text  = (item.get("text")  or "").strip()
        if title and text:
            rows.append(f"{title}\n{text}")
        elif text:
            rows.append(text)
        elif title:
            rows.append(title)
    print(f"  Financial news: {len(rows):,} articles")

    # ── Finance-Alpaca (formatted as plain instruct text) ───────────────────
    print("Loading gbharti/finance-alpaca …")
    ds_alpaca = load_dataset("gbharti/finance-alpaca", split="train")
    alpaca_count = 0
    for item in ds_alpaca:
        inst = (item.get("instruction") or "").strip()
        out  = (item.get("output")      or "").strip()
        ctx  = (item.get("input")       or "").strip()
        if not inst or not out:
            continue
        parts = [f"### Instruction:\n{inst}"]
        if ctx:
            parts.append(f"### Input:\n{ctx}")
        parts.append(f"### Response:\n{out}")
        rows.append("\n\n".join(parts))
        alpaca_count += 1
    print(f"  Finance-Alpaca: {alpaca_count:,} examples")

    random.shuffle(rows)
    print(f"  Total finance domain mix: {len(rows):,} rows")
    cache_path = Path(cache_dir) / f"finance_domain_mix_gpt2_{vocab_size}_{_CACHE_VERSION}.pt"
    dataset = TextDataset(rows, vocab_size, seq_len, batch_size, device, cache_path)
    del rows
    return dataset


def build_finance_alpaca(
    vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str
) -> SFTDataset:
    """Phase 3: gbharti/finance-alpaca — 21K financial instruction examples (response-only loss)."""
    from datasets import load_dataset
    print("Loading gbharti/finance-alpaca …")
    ds = load_dataset("gbharti/finance-alpaca", split="train")
    pairs = _extract_sft_pairs(ds, "finance-alpaca",
                               instruction_cols=["instruction"],
                               output_cols=["output"],
                               input_cols=["input"])
    cache_path = Path(cache_dir) / f"finance_alpaca_sft_{vocab_size}_{_CACHE_VERSION}.pt"
    return SFTDataset(pairs, vocab_size, seq_len, batch_size, device, cache_path)


def build_trading_qa(
    vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str
) -> SFTDataset:
    """Phase 5: FinGPT/fingpt-fiqa_qa — risk-management QA, final calibration phase."""
    from datasets import load_dataset
    print("Loading FinGPT/fingpt-fiqa_qa …")
    ds = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
    pairs = _extract_sft_pairs(ds, "fingpt-fiqa_qa",
                               instruction_cols=["instruction"],
                               output_cols=["output", "answer", "response"])
    cache_path = Path(cache_dir) / f"trading_qa_sft_{vocab_size}_{_CACHE_VERSION}.pt"
    return SFTDataset(pairs, vocab_size, seq_len, batch_size, device, cache_path)


def build_trading_methodology_sft(
    vocab_size: int, seq_len: int, batch_size: int, device: torch.device, cache_dir: str
) -> SFTDataset:
    """Phase 4: fingpt-forecaster-dow30 + fingpt-sentiment-train combined (~78K pairs).

    Forecaster teaches market analysis and directional prediction (core of trading method).
    Sentiment teaches reading market signals from news (input to trading decisions).
    Runs BEFORE the risk-QA phase so the final calibration can anchor risk/uncertainty tone.

    Dataset path changed from FinGPT/fingpt-forecaster (deprecated, 401) to the versioned
    Dow 30 dataset FinGPT/fingpt-forecaster-dow30-202305-202405 (publicly accessible).
    """
    from datasets import load_dataset
    print("Loading Phase 4 trading methodology datasets …")

    pairs = []

    ds_fore = load_dataset("FinGPT/fingpt-forecaster-dow30-202305-202405", split="train")
    fore = _extract_sft_pairs(ds_fore, "fingpt-forecaster-dow30",
                              instruction_cols=["prompt", "instruction"],
                              output_cols=["answer", "output"],
                              input_cols=["input"])
    print(f"  fingpt-forecaster-dow30: {len(fore):,} examples")
    pairs.extend(fore)

    ds_sent = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    sent = _extract_sft_pairs(ds_sent, "fingpt-sentiment-train",
                              instruction_cols=["instruction"],
                              output_cols=["output", "answer", "label"],
                              input_cols=["input"])
    print(f"  fingpt-sentiment-train: {len(sent):,} examples")
    pairs.extend(sent)

    print(f"  Total Phase 4: {len(pairs):,} examples")
    cache_path = Path(cache_dir) / f"trading_methodology_sft_{vocab_size}_{_CACHE_VERSION}.pt"
    return SFTDataset(pairs, vocab_size, seq_len, batch_size, device, cache_path)


# ──────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────

def make_optimizer(params, lr: float, optim8bit: bool = False):
    """AdamW を作る。optim8bit=True かつ bitsandbytes/CUDA が使えれば 8-bit Adam。

    8-bit Adam はオプティマイザ状態(m, v)を block-wise に 8-bit 量子化し、
    fp32 の 8 byte/param を ~2 byte/param に削減する(ほぼ無損失)。使えない環境
    (CUDA/bitsandbytes 無し)では通常 AdamW に安全にフォールバックする。
    """
    kw = dict(lr=lr, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    if optim8bit:
        try:
            import bitsandbytes as bnb
            if not torch.cuda.is_available():
                raise RuntimeError("bitsandbytes 8-bit optimizer は CUDA が必要です")
            print("Optimizer: bitsandbytes AdamW8bit (optimizer states in 8-bit)")
            return bnb.optim.AdamW8bit(params, **kw)
        except Exception as e:
            print(f"  [warn] 8-bit optimizer 利用不可 ({e}); 通常 AdamW にフォールバック")
    print("Optimizer: torch.optim.AdamW (fp32 states)")
    return AdamW(params, **kw)


def lr_lambda(step: int, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ──────────────────────────────────────────────────────────────
# Loop curriculum (experimental, script-driven)
# ──────────────────────────────────────────────────────────────
# モデル内部にも素朴な loop_curriculum（毎 step randint(1, max)）があるが、
# ここではフェーズ別レンジ + max を超える裾を学習スクリプト側から明示制御する。
# n_loops を明示指定するとモデル内部のサンプリングは上書きされる。

def phase_loop_range(phase_idx: int, progress: float) -> tuple[int, int]:
    """フェーズと進捗(0..1)から (lo, hi) の基本レンジを返す。

    合意したスケジュール:
      Phase 1 前半: 1–4 / Phase 1 後半: 2–8 / Phase 2 以降: 4–8
    """
    if phase_idx <= 1:
        return (1, 4) if progress < 0.5 else (2, 8)
    return (4, 8)


def sample_curriculum_loops(phase_idx: int, progress: float, step: int,
                            args: argparse.Namespace) -> int:
    """学習ステップごとに n_loops をサンプリングする（resume 安全な決定的シード）。

    確率 tail_p で (hi+1 .. tail_max) の「裾」を引き、depth extrapolation を学習させる。
    """
    lo, hi = phase_loop_range(phase_idx, progress)
    # step+seed のみに依存する決定的 RNG（resume しても同じ系列を再現）
    rng = random.Random(args.loop_seed * 1_000_003 + step)
    # 裾(>hi)は Phase 2 以降のみ（Phase 1 は浅いウォームアップに専念させる）
    if phase_idx >= 2 and args.loop_tail_max > hi and rng.random() < args.loop_tail_p:
        return rng.randint(hi + 1, args.loop_tail_max)
    return rng.randint(lo, hi)


def _phase_idx_from_name(phase_name: str) -> int:
    """'Phase3-FinanceDomain' → 3。失敗時は 0。"""
    m = re.search(r"[Pp]hase\s*(\d)", phase_name)
    return int(m.group(1)) if m else 0


def _curriculum_ramp(progress: float, start: float, end: float,
                     warmup_frac: float) -> float:
    """全学習進捗 progress∈[0,1] に対し start→end を線形ランプし、以降 end で保持。

    warmup_frac は「全学習のうち何割でランプを完了するか」(0<frac≤1)。
    例: start=0.5, end=0.99, warmup_frac=0.5 なら前半50%で 0.5→0.99 に上げ、後半は 0.99 固定。
    warmup_frac<=0 のときは常に end を返す(ランプ無し=即 end)。
    """
    if warmup_frac <= 0.0:
        return end
    frac = min(max(progress, 0.0) / warmup_frac, 1.0)
    return start + (end - start) * frac


def validate_act_curriculum_args(args: argparse.Namespace) -> None:
    """ACT カリキュラム CLI の値域を検証する(不正なら ValueError)。

    - act_threshold_start/end: (0, 1]  — ACT 停止確率の累積閾値
    - act_warmup_frac: [0, 1]
    - ponder_weight_start/end: >= 0  — 負値は余分ループを報酬化してしまう
    """
    for nm in ("act_threshold_start", "act_threshold_end"):
        v = getattr(args, nm)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"--{nm} は (0, 1] の範囲で指定してください (got {v})。"
                             "ACT 停止確率の累積閾値です。")
    if not (0.0 <= args.act_warmup_frac <= 1.0):
        raise ValueError(f"--act_warmup_frac は [0, 1] で指定してください (got {args.act_warmup_frac})。")
    for nm in ("ponder_weight_start", "ponder_weight_end"):
        v = getattr(args, nm)
        if v < 0.0:
            raise ValueError(f"--{nm} は >= 0 で指定してください (got {v})。"
                             "負値は余分なループを報酬化してしまいます。")


def apply_act_curriculum(model, args: argparse.Namespace,
                         step: int, grand_total: int,
                         anchor_step: int = 0) -> tuple[float, float]:
    """ACT カリキュラム: 学習進捗に応じて act_threshold / ponder weight を更新。

    モデルは forward 時に毎回 cfg を読むため、共有 cfg オブジェクトを書き換えるだけで
    次の forward から反映される(モデルコード非改変)。返り値は現在値(ログ用)。

    進捗は anchor_step(今回の学習が始まった step)から grand_total までの「今回学習する
    区間」で測る。phase1_final から resume して phase2-5 のみ学習する場合でも、その区間で
    start→end をランプできる(anchor_step=0 なら全フェーズ通しの挙動)。

    - act_threshold: start→end へ線形に「上げる」と、初期は浅いループで早期停止し、
      後半ほど深い推論を解禁する(浅→深カリキュラム)。
    - act_aux_loss_weight(ponder cost): start→end へ「下げる」と、初期は余分なループに
      強くペナルティを掛け、後半で緩める。閾値ランプと同方向(浅→深)に効く補助レバー。
    """
    span = max(grand_total - 1 - anchor_step, 1)
    progress = (step - anchor_step) / span
    thr = _curriculum_ramp(progress, args.act_threshold_start,
                           args.act_threshold_end, args.act_warmup_frac)
    pon = _curriculum_ramp(progress, args.ponder_weight_start,
                           args.ponder_weight_end, args.act_warmup_frac)
    model.cfg.act_threshold = thr
    model.cfg.act_aux_loss_weight = pon
    return thr, pon


# ──────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix added by torch.compile so checkpoints are portable."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                for k, v in state_dict.items()}
    return state_dict


def _safe_torch_load(path: str, allow_unsafe: bool = False) -> dict:
    """Load a checkpoint with weights_only=True by default.

    PyTorch checkpoints are pickle-based; weights_only=False allows arbitrary
    code execution if the file is malicious. Use allow_unsafe=True only for
    checkpoints you created yourself or fully trust.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_err:
        if not allow_unsafe:
            raise RuntimeError(
                f"Cannot load {path!r} with weights_only=True: {first_err}\n"
                "If this is a checkpoint you created yourself, re-run with "
                "--allow_unsafe_checkpoint to permit pickle-based loading."
            ) from first_err
        import warnings
        warnings.warn(
            f"weights_only=True failed for {path!r}: {first_err}. "
            "Falling back to weights_only=False — only load trusted checkpoints this way.",
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(
    path: Path,
    step: int,
    model,
    optimizer,
    scheduler,
    cfg: MythosConfig,
    tag: str = "",
    phase1_steps: int = 0,
    phase2_steps: int = 0,
    phase3_steps: int = 0,
    phase4_steps: int = 0,
    phase5_steps: int = 0,
    scaler=None,
):
    payload = {
        "step": step,
        "model_state": _strip_compile_prefix(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_type": type(optimizer).__name__,  # resume 時の不一致検出用
        "scheduler_state": scheduler.state_dict(),
        "cfg": cfg.__dict__,
        "tag": tag,
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "phase3_steps": phase3_steps,
        "phase4_steps": phase4_steps,
        "phase5_steps": phase5_steps,
    }
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state"] = scaler.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  → Saved: {path}")


def _optimizer_state_compatible(optimizer, saved_state) -> bool:
    """保存された optimizer state が現在の optimizer と構造的に互換か判定する。

    bitsandbytes の 8-bit optimizer は per-param state に 'state1'/'state2' を持ち、
    torch の AdamW は 'exp_avg'/'exp_avg_sq' を持つ。両者を混在ロードすると
    load_state_dict 自体は通るが step 時に KeyError('state1') 等で壊れるため、
    optimizer_type が記録されていない旧 checkpoint でも構造から弾けるようにする。
    """
    sample_keys: set = set()
    for s in (saved_state.get("state") or {}).values():
        sample_keys = set(s.keys())
        break
    if not sample_keys:
        return True  # state 空(未ステップ)なら何にでもロード可
    is_8bit = "bitsandbytes" in type(optimizer).__module__.lower()
    saved_is_8bit = "state1" in sample_keys
    saved_is_fp32 = "exp_avg" in sample_keys
    if is_8bit and saved_is_fp32:
        return False  # fp32 state → 8-bit optimizer
    if (not is_8bit) and saved_is_8bit:
        return False  # 8-bit state → fp32 optimizer
    return True


def load_checkpoint(path: str, model, optimizer, scheduler, scaler=None, allow_unsafe: bool = False):
    print(f"Resuming from: {path}")
    ckpt = _safe_torch_load(path, allow_unsafe=allow_unsafe)
    state_key = "model_state" if "model_state" in ckpt else "model"
    ckpt_sd = _strip_compile_prefix(ckpt[state_key])

    # If model is torch.compile()'d, its state_dict keys have _orig_mod. prefix
    model_sd = model.state_dict()
    model_keys = list(model_sd.keys())
    if model_keys and model_keys[0].startswith("_orig_mod."):
        ckpt_sd = {"_orig_mod." + k: v for k, v in ckpt_sd.items()}

    # Filter shape-mismatched keys (e.g. freqs_cis when seq_len changes)
    filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in ckpt_sd if k not in filtered]
    if skipped:
        print(f"  Skipping shape-mismatched keys: {skipped}")
    model.load_state_dict(filtered, strict=False)
    if "optimizer_state" in ckpt:
        # optimizer 種別が違う（例: fp32 AdamW ⇄ 8-bit AdamW8bit）と state 構造が
        # 合わず壊れる。不一致/失敗時は警告してリセット（model/scheduler は継続）。
        saved_optim = ckpt.get("optimizer_type")
        cur_optim = type(optimizer).__name__
        type_mismatch = saved_optim is not None and saved_optim != cur_optim
        # optimizer_type が無い旧 checkpoint でも state 構造から非互換を検出する
        # (例: fp32 AdamW state を --optim8bit の AdamW8bit にロード → step で KeyError)
        struct_incompatible = not _optimizer_state_compatible(optimizer, ckpt["optimizer_state"])
        if type_mismatch or struct_incompatible:
            why = (f"種別不一致 (保存={saved_optim} / 現在={cur_optim})" if type_mismatch
                   else "state 構造が現在の optimizer と非互換 (8-bit ⇄ fp32)")
            print(f"  [warn] optimizer {why}。optimizer state を読み込まずリセットします"
                  "（resume では --optim8bit の有無を揃えると momentum も継続できます）。")
        else:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as _oe:  # noqa: BLE001
                print(f"  [warn] optimizer state の読込に失敗 ({_oe})。リセットして継続します。")
    if "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    else:
        # Legacy checkpoints without scheduler_state: replay steps
        for _ in range(ckpt["step"]):
            scheduler.step()
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step = ckpt["step"]
    p1 = ckpt.get("phase1_steps", "?")
    p2 = ckpt.get("phase2_steps", "?")
    p3 = ckpt.get("phase3_steps", "?")
    p4 = ckpt.get("phase4_steps", "?")
    p5 = ckpt.get("phase5_steps", "?")
    print(f"  Resumed at step {step}  tag={ckpt.get('tag', '')}")
    known = [x for x in [p1, p2, p3, p4, p5] if x != "?"]
    if known:
        print(f"  NOTE: use --phase1_steps {p1} --phase2_steps {p2} "
              f"--phase3_steps {p3} --phase4_steps {p4} --phase5_steps {p5} to match the original run.")
    return step


# ──────────────────────────────────────────────────────────────
# Single training phase
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# Memory replay (rehearsal) — anti-forgetting
# ──────────────────────────────────────────────────────────────
# CLS（補完学習系）的に、後フェーズの学習中に前フェーズ（一般言語）の
# バッチを少量インターリーブして破滅的忘却を抑える。

def _cycle_batches(dataset):
    """データセットを無限に巡回してバッチを供給する（リプレイ用）。"""
    while True:
        for b in dataset:
            yield b


def run_phase(
    phase_name: str,
    dataset,
    model: BushidoMythos,
    cfg: MythosConfig,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    ckpt_dir: Path,
    start_step: int,
    total_steps: int,
    phase_final_name: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    phase1_steps: int = 0,
    phase2_steps: int = 0,
    phase3_steps: int = 0,
    phase4_steps: int = 0,
    phase5_steps: int = 0,
    resume_path: str = None,
    replay_dataset=None,
    act_anchor_step: int = 0,
) -> int:
    total_loss = 0.0
    total_ce   = 0.0
    total_aux  = 0.0
    total_loops = 0   # サンプリングした n_loops の累積（curriculum 検証・速度効果の確認用）
    n_loops_micros = 0
    total_replay = 0  # リプレイで差し替えた micro-batch 数（忘却対策の発火確認用）
    step = start_step
    t0 = time.time()

    # loop curriculum モード解決
    phase_idx = _phase_idx_from_name(phase_name)
    loop_mode = getattr(args, "loop_schedule", "off")

    # ACT カリキュラム: 全フェーズ合計ステップを「学習全体」としてグローバル進捗を測る
    act_curriculum = getattr(args, "act_curriculum", False)
    act_grand_total = max(
        phase1_steps + phase2_steps + phase3_steps + phase4_steps + phase5_steps, 1)

    # 記憶リプレイ（rehearsal）: replay_dataset があれば無限巡回イテレータを用意
    replay_ratio = getattr(args, "replay_ratio", 0.0)
    replay_iter = (_cycle_batches(replay_dataset)
                   if (replay_dataset is not None and replay_ratio > 0) else None)

    use_amp = device.type == "cuda" and amp_dtype != torch.float32
    autocast_device = device.type if device.type == "cuda" else "cpu"
    _scaler_enabled = use_amp and amp_dtype == torch.float16
    try:  # PyTorch 2.3+ prefers torch.amp.GradScaler('cuda', ...)
        scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)

    # Restore scaler state so float16 loss scaling resumes correctly
    if resume_path and _scaler_enabled:
        _ckpt = _safe_torch_load(resume_path, allow_unsafe=args.allow_unsafe_checkpoint)
        if "scaler_state" in _ckpt:
            scaler.load_state_dict(_ckpt["scaler_state"])
        del _ckpt

    grad_accum = getattr(args, 'grad_accum_steps', 1)
    eff_batch  = args.batch_size * grad_accum

    print(f"\n{'='*60}")
    print(f"  {phase_name}: steps {start_step}–{total_steps}  "
          f"batch={args.batch_size}  grad_accum={grad_accum}  eff_batch={eff_batch}  "
          f"seq_len={args.seq_len}  dtype={amp_dtype}")
    print(f"  log every {args.log_every}  save every {args.save_every}")
    if loop_mode == "curriculum":
        lo0, hi0 = phase_loop_range(phase_idx, 0.0)
        lo1, hi1 = phase_loop_range(phase_idx, 1.0)
        tail_str = (f"  tail≤{args.loop_tail_max} (p={args.loop_tail_p})"
                    if phase_idx >= 2 else "  tail=off (Phase1)")
        print(f"  loop curriculum: range {lo0}-{hi0} → {lo1}-{hi1}{tail_str}  "
              f"seed={args.loop_seed}")
    elif loop_mode == "fixed":
        print(f"  loop schedule: FIXED n_loops={cfg.max_loop_iters}")
    if replay_iter is not None:
        print(f"  memory replay: {replay_ratio*100:.0f}% of batches drawn from general-language anchor (anti-forgetting)")
    if act_curriculum:
        span = max(act_grand_total - 1 - act_anchor_step, 1)
        ramp_end_step = act_anchor_step + int(args.act_warmup_frac * span)
        print(f"  ACT curriculum: act_threshold {args.act_threshold_start}→{args.act_threshold_end}"
              f"  ponder {args.ponder_weight_start}→{args.ponder_weight_end}"
              f"  (ramp over steps {act_anchor_step}→{ramp_end_step} of {act_grand_total};"
              f" warmup_frac={args.act_warmup_frac})")
    print(f"{'='*60}\n")

    # Reset peak memory stats at phase start so the first [VRAM] log reflects
    # only training-time usage, not model load / compile / initialization peaks.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    log_micros = 0  # actual micro-batch count since last log reset (avoids skewed average on resume)
    epoch = 0
    while step < total_steps:
        epoch += 1
        for batch in dataset:
            if step >= total_steps:
                break

            # 記憶リプレイ: 確率 replay_ratio で一般言語(WikiText)アンカーに差し替え。
            # 決定は step ベースの決定的 RNG（resume 安全・loop_seed と同思想）。
            # step は grad_accum グループ内で一定なので、グループ単位で replay/通常が揃う。
            # loop curriculum (×1_000_003) と別乗数でデコリレート。
            if replay_iter is not None:
                _rrng = random.Random(getattr(args, "loop_seed", 0) * 2_654_435_761 + step)
                if _rrng.random() < replay_ratio:
                    batch = next(replay_iter)
                    total_replay += 1

            # SFTDataset yields (x, y, loss_mask); TextDataset yields (x, y)
            if len(batch) == 3:
                x, y, loss_mask = batch
                if not loss_mask.any():
                    continue  # chunk is prompt-only — no response tokens to learn from
            else:
                x, y = batch
                loss_mask = None

            # n_loops の決定（off=モデル既定 / fixed=max固定 / curriculum=フェーズ別+裾）
            if loop_mode == "fixed":
                n_loops = cfg.max_loop_iters
            elif loop_mode == "curriculum":
                denom = max(total_steps - start_step, 1)
                progress = (step - start_step) / denom
                n_loops = sample_curriculum_loops(phase_idx, progress, step, args)
            else:  # "off": モデル内部のロジック（cfg.loop_curriculum）に委ねる
                n_loops = None
            if n_loops is not None:
                total_loops += n_loops
                n_loops_micros += 1

            # ACT カリキュラム: グローバル進捗に応じて act_threshold / ponder を更新
            # (forward は毎回 cfg を読むので、ここでの書き換えが次の forward に効く)
            if act_curriculum:
                apply_act_curriculum(model, args, step, act_grand_total,
                                     anchor_step=act_anchor_step)

            with torch.autocast(autocast_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(x, n_loops=n_loops)
                if loss_mask is not None and loss_mask.any():
                    ce_per_tok = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        y.reshape(-1),
                        reduction="none",
                    )
                    mask_f = loss_mask.reshape(-1).float()
                    ce_loss = (ce_per_tok * mask_f).sum() / (mask_f.sum() + 1e-9)
                else:
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        y.reshape(-1),
                    )
                loss = ce_loss + model._last_aux_loss

            # Accumulate loss stats for logging (unscaled, per-micro-batch average)
            total_loss += loss.item()
            total_ce   += ce_loss.item()
            total_aux  += model._last_aux_loss.item()
            log_micros += 1

            # Divide by grad_accum so summed gradients equal one full-batch gradient
            scaler.scale(loss / grad_accum).backward()
            micro_step += 1

            if micro_step % grad_accum != 0:
                continue  # keep accumulating micro-batch gradients

            # ── Optimizer update (once per grad_accum micro-batches) ───────
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            model.update_moe_router_bias()
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                n = max(log_micros, 1)
                avg_loss = total_loss / n
                avg_ce   = total_ce   / n
                avg_aux  = total_aux  / n
                ppl = math.exp(min(avg_ce, 20))
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                loops_str = ""
                if n_loops_micros > 0:
                    loops_str = f"  loops≈{total_loops / n_loops_micros:.1f}"
                replay_str = ""
                if replay_iter is not None and log_micros > 0:
                    replay_str = f"  replay={total_replay / max(log_micros,1)*100:.0f}%"
                act_str = ""
                if act_curriculum:
                    act_str = f"  act_thr={model.cfg.act_threshold:.3f}"
                    if model.cfg.act_aux_loss_weight:
                        act_str += f"  ponder={model.cfg.act_aux_loss_weight:.3f}"
                print(
                    f"[{phase_name}] step {step:6d}/{total_steps}"
                    f"  loss={avg_loss:.4f}  ce={avg_ce:.4f}  aux={avg_aux:.4f}"
                    f"  ppl={ppl:.1f}  lr={lr_now:.2e}{loops_str}{replay_str}{act_str}  elapsed={elapsed:.0f}s"
                )
                total_loss = total_ce = total_aux = 0.0
                total_loops = 0
                n_loops_micros = 0
                total_replay = 0
                log_micros = 0
                t0 = time.time()

            mem_every = getattr(args, "mem_log_every", 100)
            if mem_every > 0 and step % mem_every == 0:
                vram = _vram_str(device, reset_peak=True)
                if vram:
                    print(f"[VRAM] step {step:6d}  {vram}")

            if step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:06d}.pt",
                    step, model, optimizer, scheduler, cfg, tag=phase_name,
                    phase1_steps=phase1_steps, phase2_steps=phase2_steps,
                    phase3_steps=phase3_steps, phase4_steps=phase4_steps,
                    phase5_steps=phase5_steps, scaler=scaler,
                )

    final_path = ckpt_dir / phase_final_name
    save_checkpoint(
        final_path, step, model, optimizer, scheduler, cfg, tag=phase_name,
        phase1_steps=phase1_steps, phase2_steps=phase2_steps,
        phase3_steps=phase3_steps, phase4_steps=phase4_steps,
        phase5_steps=phase5_steps, scaler=scaler,
    )
    print(f"\n{phase_name} complete. Checkpoint: {final_path}\n")
    return step, scaler


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # 学習全体の RNG seed を起動直後に固定。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # batch 順序は専用 Generator(TextDataset/SFTDataset._gen)で決める。これをモデル構築の
    # torch global RNG 消費から分離することが肝心: MLA/GQA は構造が違い RNG 消費量も違うため、
    # global RNG 依存のままだと randperm の batch 順序が別プロセス間でずれる。
    # データセットは _DATA_SAMPLE_SEED を読むので、ここで args.seed に揃える。
    global _DATA_SAMPLE_SEED
    _DATA_SAMPLE_SEED = args.seed

    device, amp_dtype = get_device_and_dtype(args.dtype)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else ckpt_dir / "train.log"
    tee = _Tee(log_path)
    print(f"Logging to: {log_path}")
    print(f"Device: {device}  dtype: {amp_dtype}")

    base_ckpt_path = args.base_ckpt

    # ── Determine resume path first (needed for config fallback) ─
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        candidates = sorted(ckpt_dir.glob("step_*.pt"))
        if candidates:
            resume_path = str(candidates[-1])

    # ── Load config ───────────────────────────────────────────
    # Priority: resume checkpoint > base_ckpt > error
    base_ckpt = None
    if Path(base_ckpt_path).exists():
        print(f"Loading base checkpoint: {base_ckpt_path}")
        base_ckpt = _safe_torch_load(base_ckpt_path, allow_unsafe=args.allow_unsafe_checkpoint)
        cfg = MythosConfig(**base_ckpt["cfg"])
    elif resume_path and Path(resume_path).exists():
        print(f"Base checkpoint not found. Borrowing config from: {resume_path}")
        _tmp = _safe_torch_load(resume_path, allow_unsafe=args.allow_unsafe_checkpoint)
        cfg = MythosConfig(**_tmp["cfg"])
        del _tmp
    else:
        raise FileNotFoundError(
            f"No checkpoint found.\n"
            f"  --base_ckpt: {base_ckpt_path} (not found)\n"
            f"  --ckpt_dir:  {ckpt_dir} (no step_*.pt)\n"
            f"Provide --base_ckpt or ensure --ckpt_dir has at least one step_*.pt."
        )

    # Apply seq_len override
    if args.seq_len is not None:
        cfg = MythosConfig(**{**cfg.__dict__, "max_seq_len": args.seq_len})

    # Apply gradient checkpointing override (always force from CLI arg to avoid
    # accidentally inheriting a stale True from a checkpoint's cfg)
    if args.grad_checkpoint != cfg.use_gradient_checkpointing:
        cfg = MythosConfig(**{**cfg.__dict__, "use_gradient_checkpointing": args.grad_checkpoint})

    model = BushidoMythos(cfg).to(device)

    # ── Load base weights (skipped when resuming — resume overwrites them) ─
    if base_ckpt is not None and resume_path is None:
        state_key = "model_state" if "model_state" in base_ckpt else "model"
        ckpt_sd = base_ckpt[state_key]
        model_sd = model.state_dict()
        filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in ckpt_sd if k not in filtered]
        if skipped:
            print(f"  Skipping shape-mismatched keys (will be re-init): {skipped}")
        missing, _ = model.load_state_dict(filtered, strict=False)
        trainable = {n for n, _ in model.named_parameters()}
        missing_trained = [k for k in missing if k in trainable]
        if missing_trained:
            print(f"  WARNING: missing trained params: {missing_trained}")
    elif base_ckpt is not None and resume_path is not None:
        print("  Base checkpoint found but resume takes priority — skipping base weights.")
    else:
        print("  No base checkpoint — resume checkpoint will supply all weights.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  ({n_params/1e6:.1f}M)  vocab={cfg.vocab_size:,}")

    # ── ACT curriculum: 終了閾値の既定(-1)は cfg.act_threshold を採用 ──
    if args.act_curriculum and args.act_threshold_end < 0:
        args.act_threshold_end = cfg.act_threshold

    # ── ACT curriculum: 値域チェック ──
    if args.act_curriculum:
        validate_act_curriculum_args(args)

    # ── ACT curriculum × torch.compile は両立しない(暫定対応) ──
    # apply_act_curriculum が毎ステップ Python 属性 cfg.act_threshold を書き換え、
    # forward 内で読むため、torch.compile の guard 再評価で再コンパイルが多発する。
    # 当面は compile を自動無効化して安全側に倒す(恒久対応は閾値の tensor buffer 化)。
    if args.act_curriculum and args.compile:
        print("  [warn] --act_curriculum と --compile は両立しません(閾値変更ごとに再コンパイル)。"
              "compile を無効化して継続します。高速化が必要なら act_threshold を固定して compile を使うか、"
              "閾値の tensor buffer 化(恒久対応)を待ってください。")
        args.compile = False

    # ── Phase step totals ─────────────────────────────────────
    p1_total = args.phase1_steps
    p2_total = p1_total + args.phase2_steps
    p3_total = p2_total + args.phase3_steps
    p4_total = p3_total + args.phase4_steps
    p5_total = p4_total + args.phase5_steps

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = make_optimizer(model.parameters(), args.lr,
                               optim8bit=getattr(args, "optim8bit", False))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_lambda(step, args.warmup_steps, p5_total),
    )

    # ── Resume ────────────────────────────────────────────────
    step = 0
    if resume_path:
        step = load_checkpoint(resume_path, model, optimizer, scheduler,
                               allow_unsafe=args.allow_unsafe_checkpoint)

    # ACT カリキュラムのランプ起点。既定(-1)は今回の resume step を自動採用。
    # フェーズを別プロセスで分割実行する場合は --act_anchor_step に固定値(=phase1 合計
    # step)を渡すことで、別プロセスをまたいで連続ランプにできる(各 resume step で
    # 起点がリセットされるのを防ぐ)。
    act_anchor_step = args.act_anchor_step if args.act_anchor_step >= 0 else step
    if args.act_curriculum:
        print(f"[ACT curriculum] anchor_step={act_anchor_step} "
              f"({'explicit' if args.act_anchor_step >= 0 else 'auto=resume step'})")

    # ── torch.compile (after all checkpoint loading) ──────────
    if args.compile:
        print("Compiling model with torch.compile() …")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"  torch.compile() failed ({e}); continuing without compile.")

    last_scaler = None  # tracks scaler from the last run_phase for final.pt

    # ── Memory replay anchor (general-language / WikiText-103) for Phase 2+ ──
    # 破滅的忘却対策。アンカーは「前フェーズ全混合」ではなく一般言語(WikiText)のみ
    # = general-language replay。Phase 5 の汎用劣化(PPL 54→361)を直撃する設計。
    # 構築は Phase 2 直前に遅延（Phase 1 を回した場合は ds1 を再利用し WikiText を二重に持たない）。
    replay_ds = None

    # ── Phase 1: WikiText-103 ─────────────────────────────────
    if args.phase in (0, 1) and step < p1_total:
        print("\n[Phase 1] Building WikiText-103 dataset …")
        ds1 = build_wikitext103(cfg.vocab_size, args.seq_len, args.batch_size, device, args.cache_dir)
        step, last_scaler = run_phase(
            phase_name="Phase1-WikiText103",
            dataset=ds1,
            model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
            args=args, ckpt_dir=ckpt_dir, start_step=step, total_steps=p1_total,
            phase_final_name="phase1_final.pt", device=device, amp_dtype=amp_dtype,
            phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
            phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
            phase5_steps=args.phase5_steps, resume_path=resume_path,
            act_anchor_step=act_anchor_step,
        )
        if args.replay_ratio > 0:
            replay_ds = ds1  # 一般言語アンカーを再利用（WikiText を二重に保持しない）

    # ── Memory replay anchor: build now if Phase 1 was skipped (resume) ──
    if args.replay_ratio > 0 and replay_ds is None:
        print(f"\n[Replay] Building general-language anchor (WikiText-103) "
              f"({args.replay_ratio*100:.0f}% of Phase 2+ batches) …")
        replay_ds = build_wikitext103(cfg.vocab_size, args.seq_len, args.batch_size,
                                      device, args.cache_dir)

    # ── Phase 2: Reasoning mix (OpenWebMath + Orca Math [+ Dolly]) ──────────
    if args.phase in (0, 2) and step < p2_total:
        print("\n[Phase 2] Building reasoning mix dataset "
              f"(OpenWebMath + Orca Math{' + Dolly' if args.include_dolly else ''}) …")
        ds2 = build_reasoning_mix(
            cfg.vocab_size, args.seq_len, args.batch_size, device, args.cache_dir,
            openwebmath_rows=args.phase2_openwebmath_rows,
            orca_ratio=args.phase2_orca_ratio,
            include_dolly=args.include_dolly,
            dolly_rows=args.phase2_dolly_rows,
        )
        step, last_scaler = run_phase(
            phase_name="Phase2-ReasoningMix",
            dataset=ds2,
            model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
            args=args, ckpt_dir=ckpt_dir, start_step=step, total_steps=p2_total,
            phase_final_name="phase2_final.pt", device=device, amp_dtype=amp_dtype,
            replay_dataset=replay_ds,
            phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
            phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
            phase5_steps=args.phase5_steps, resume_path=resume_path,
            act_anchor_step=act_anchor_step,
        )

    # ── Phase 3: Finance domain mix (financial-news + finance-alpaca) ───────
    if args.phase in (0, 3) and step < p3_total:
        print("\n[Phase 3] Building finance domain mix dataset (news + alpaca) …")
        ds3 = build_finance_domain_mix(cfg.vocab_size, args.seq_len, args.batch_size, device, args.cache_dir)
        step, last_scaler = run_phase(
            phase_name="Phase3-FinanceDomain",
            dataset=ds3,
            model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
            args=args, ckpt_dir=ckpt_dir, start_step=step, total_steps=p3_total,
            phase_final_name="phase3_final.pt", device=device, amp_dtype=amp_dtype,
            replay_dataset=replay_ds,
            phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
            phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
            phase5_steps=args.phase5_steps, resume_path=resume_path,
            act_anchor_step=act_anchor_step,
        )

    # ── Phase 4: Trading methodology SFT (forecaster + sentiment) ─
    if args.phase in (0, 4) and step < p4_total:
        print("\n[Phase 4] Building trading methodology dataset (forecaster + sentiment) …")
        ds4 = build_trading_methodology_sft(cfg.vocab_size, args.seq_len, args.batch_size, device, args.cache_dir)
        step, last_scaler = run_phase(
            phase_name="Phase4-TradingMethodology",
            dataset=ds4,
            model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
            args=args, ckpt_dir=ckpt_dir, start_step=step, total_steps=p4_total,
            phase_final_name="phase4_final.pt", device=device, amp_dtype=amp_dtype,
            replay_dataset=replay_ds,
            phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
            phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
            phase5_steps=args.phase5_steps, resume_path=resume_path,
            act_anchor_step=act_anchor_step,
        )

    # ── Phase 5: Trading discipline / risk-management QA ─────────
    if args.phase in (0, 5) and step < p5_total:
        print("\n[Phase 5] Building trading risk-management QA dataset …")
        ds5 = build_trading_qa(cfg.vocab_size, args.seq_len, args.batch_size, device, args.cache_dir)
        step, last_scaler = run_phase(
            phase_name="Phase5-TradingQA",
            dataset=ds5,
            model=model, cfg=cfg, optimizer=optimizer, scheduler=scheduler,
            args=args, ckpt_dir=ckpt_dir, start_step=step, total_steps=p5_total,
            phase_final_name="phase5_final.pt", device=device, amp_dtype=amp_dtype,
            replay_dataset=replay_ds,
            phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
            phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
            phase5_steps=args.phase5_steps, resume_path=resume_path,
            act_anchor_step=act_anchor_step,
        )

    # ── Final checkpoint ──────────────────────────────────────
    final_path = ckpt_dir / "final.pt"
    save_checkpoint(
        final_path, step, model, optimizer, scheduler, cfg, tag="finance_final",
        phase1_steps=args.phase1_steps, phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps, phase4_steps=args.phase4_steps,
        phase5_steps=args.phase5_steps, scaler=last_scaler,
    )
    print(f"All done. Final model: {final_path}")
    tee.close()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="5-phase finance pretraining: WikiText-103 → reasoning mix → finance domain → trading methodology → risk-management QA"
    )
    # Phase control
    p.add_argument("--phase",         type=int,   default=0,
                   choices=[0, 1, 2, 3, 4, 5],
                   help="0=all phases, 1=WikiText-103, 2=reasoning mix (OpenWebMath+OrcaMath+Dolly), "
                        "3=finance domain (news+alpaca), 4=trading methodology (forecaster+sentiment), "
                        "5=risk-management QA (final calibration)")
    p.add_argument("--phase1_steps",  type=int,   default=20_000,
                   help="Steps for phase 1 (WikiText-103 general fluency)")
    p.add_argument("--phase2_steps",  type=int,   default=8_000,
                   help="Steps for phase 2 (reasoning mix: OpenWebMath + Orca Math [+ Dolly])")
    p.add_argument("--phase2_openwebmath_rows", type=int, default=80_000,
                   help="OpenWebMath rows to stream for Phase 2 (full dataset = 6.3B tokens)")
    p.add_argument("--phase2_orca_ratio", type=float, default=35.0,
                   help="Orca Math rows as %% of openwebmath_rows (default 35 → ~47K rows)")
    p.add_argument("--include_dolly",  action="store_true",
                   help="Include databricks-dolly-15k in Phase 2 reasoning mix. "
                        "License: CC BY-SA 3.0 — enable only when compatible with your use-case.")
    p.add_argument("--phase2_dolly_rows", type=int, default=15_000,
                   help="Dolly rows cap when --include_dolly is set (default 15000)")
    p.add_argument("--phase3_steps",  type=int,   default=8_000,
                   help="Steps for phase 3 (finance domain: financial-news + finance-alpaca plain text)")
    p.add_argument("--phase4_steps",  type=int,   default=3_000,
                   help="Steps for phase 4 (trading methodology: forecaster-dow30 ~1.2K + sentiment ~76K SFT)")
    p.add_argument("--phase5_steps",  type=int,   default=3_000,
                   help="Steps for phase 5 (trading discipline / risk-management QA — final calibration)")

    # Data / model
    p.add_argument("--base_ckpt",     type=str,
                   default="checkpoints/a100_v2_gpt2vocab/final.pt",
                   help="Starting checkpoint (model config + weights)")
    p.add_argument("--seq_len",       type=int,   default=1024,
                   help="Sequence length (longer = richer context, more RAM)")
    p.add_argument("--batch_size",    type=int,   default=4,
                   help="Micro-batch size per accumulation step")
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size × grad_accum_steps")
    p.add_argument("--cache_dir",     type=str,   default=".cache",
                   help="Directory for cached tokenised tensors")
    p.add_argument("--ckpt_dir",      type=str,   default="checkpoints/finance_a100_v2",
                   help="Output checkpoint directory")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true",
                   help="Allow weights_only=False when loading checkpoints (pickle-based; "
                        "only use with checkpoints you created yourself)")

    # Optimiser
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Peak learning rate (lower than from-scratch to avoid forgetting)")
    p.add_argument("--warmup_steps",  type=int,   default=200,
                   help="LR warm-up steps")
    p.add_argument("--grad_clip",     type=float, default=1.0,
                   help="Gradient clipping norm")

    # Logging / checkpointing
    p.add_argument("--mem_log_every", type=int,   default=100,
                   help="Log VRAM stats (alloc/reserved/peak/frag) every N steps. 0 = disable.")
    p.add_argument("--log_every",     type=int,   default=200,
                   help="Log frequency (steps)")
    p.add_argument("--save_every",    type=int,   default=2000,
                   help="Checkpoint frequency (steps)")
    p.add_argument("--log_file",      type=str,   default=None,
                   help="Path for the training log file (default: <ckpt_dir>/train.log)")

    # Resume
    p.add_argument("--resume",        type=str,   default=None,
                   help="Explicit checkpoint to resume from (overrides --auto_resume)")
    p.add_argument("--auto_resume",   action="store_true",
                   help="Auto-resume from latest step_*.pt in --ckpt_dir")

    # GPU acceleration
    p.add_argument("--dtype",         type=str,   default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"],
                   help="Training dtype: auto=bfloat16 on Ampere+(A100), float16 on older CUDA(T4), float32 on CPU/MPS")
    p.add_argument("--compile",       action="store_true",
                   help="Apply torch.compile() for ~20-40%% additional GPU speedup (first step takes ~60s to compile)")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Enable gradient checkpointing in the recurrent loop. "
                        "Reduces VRAM ~proportionally to loop depth (e.g. 7x for 8 loops) "
                        "at the cost of ~30-40%% extra compute. Recommended when OOM.")
    p.add_argument("--optim8bit", action="store_true",
                   help="bitsandbytes の 8-bit AdamW を使う（オプティマイザ状態を 8-bit 量子化、"
                        "8→2 byte/param に削減・ほぼ無損失）。CUDA/bitsandbytes が無い場合は"
                        "通常 AdamW に自動フォールバック")

    # Loop curriculum (experimental)
    p.add_argument("--loop_schedule", choices=["off", "fixed", "curriculum"], default="off",
                   help="再帰ループ数の制御: off=モデル既定(cfg.loop_curriculum に委譲) / "
                        "fixed=max_loop_iters 固定(クリーンな baseline) / "
                        "curriculum=フェーズ別レンジ + 裾サンプリング(実験)")
    p.add_argument("--loop_tail_max", type=int, default=12,
                   help="curriculum 時の裾の最大ループ数 (default: 12)。base ckpt の max_loop_iters 以下推奨")
    p.add_argument("--loop_tail_p", type=float, default=0.2,
                   help="curriculum 時に裾(hi+1..tail_max)を引く確率 (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="学習全体の RNG seed (default: 42)。起動直後に random/torch/cuda を "
                        "シードし、data sampling(torch.randperm)の batch 順序を決定化する。"
                        "loop sampling 用の --loop_seed とは別物。")
    p.add_argument("--loop_seed", type=int, default=0,
                   help="loop サンプラのシード (default: 0)。step+seed で決定的・resume 安全")

    # ACT curriculum (動的 act_threshold / ponder cost)
    p.add_argument("--act_curriculum", action="store_true",
                   help="ACT カリキュラムを有効化。グローバル進捗に応じて act_threshold を上げ"
                        "(浅→深)、必要なら ponder cost を下げる。既定オフ=cfg 固定値のまま。")
    p.add_argument("--act_threshold_start", type=float, default=0.5,
                   help="act_curriculum 時の開始 act_threshold (default: 0.5=浅いループで早期停止)")
    p.add_argument("--act_threshold_end", type=float, default=-1.0,
                   help="act_curriculum 時の終了 act_threshold (default: -1=cfg.act_threshold を採用)")
    p.add_argument("--act_warmup_frac", type=float, default=0.5,
                   help="全学習のうち閾値/ponder を start→end へランプし切る割合 (default: 0.5=前半で完了)")
    p.add_argument("--ponder_weight_start", type=float, default=0.0,
                   help="act_curriculum 時の開始 ponder cost (act_aux_loss_weight)。"
                        "初期に余分なループを抑えたいとき >0 に。既定 0=無効")
    p.add_argument("--ponder_weight_end", type=float, default=0.0,
                   help="act_curriculum 時の終了 ponder cost (default: 0)。start>end で後半ほど緩める")
    p.add_argument("--act_anchor_step", type=int, default=-1,
                   help="ACT カリキュラムのランプ起点 step (default: -1=今回の resume step を自動採用)。"
                        "フェーズを別プロセスで分割実行する場合、全フェーズに同じ値(=phase1 合計 step)"
                        "を渡すとランプが連続する。指定しないと各プロセスが自分の resume step を起点に"
                        "してしまい、フェーズ毎に start へリセットされる。")

    # Memory replay (anti-forgetting)
    p.add_argument("--replay_ratio", type=float, default=0.0,
                   help="記憶リプレイ: Phase 2 以降で、この割合のバッチを一般言語(WikiText-103)"
                        "アンカーに差し替える (例 0.05=5%%)。破滅的忘却(汎用性能の劣化)を抑える。"
                        "0.0=無効(既定)。注: 追加ではなく『置換』なので、値を上げると実質の"
                        "ドメイン学習量が減る — 必要なら総ステップ数を増やして補う。")

    args = p.parse_args()
    if args.grad_accum_steps < 1:
        p.error(f"--grad_accum_steps must be >= 1 (got {args.grad_accum_steps})")
    if args.loop_schedule == "curriculum":
        if args.loop_tail_max < 1:
            p.error(f"--loop_tail_max must be >= 1 (got {args.loop_tail_max})")
        if not (0.0 <= args.loop_tail_p <= 1.0):
            p.error(f"--loop_tail_p must be in [0,1] (got {args.loop_tail_p})")
    if not (0.0 <= args.replay_ratio < 1.0):
        p.error(f"--replay_ratio must be in [0,1) (got {args.replay_ratio})")
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
