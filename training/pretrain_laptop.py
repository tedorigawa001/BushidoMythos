#!/usr/bin/env python3
"""
BushidoMythos laptop pretraining — MPS / CPU対応版.

3b_fine_web_edu.py の本番品質な構造を継承しつつ:
  - FSDP / torchrun / CUDA 依存を除去
  - MPS (Apple Silicon) / CPU をサポート
  - fused AdamW を標準 AdamW に変更
  - FineWeb-Edu → WikiText / ローカル .txt 対応
  - loguru → 標準 logging
  - 任意の BushidoMythos variant を CLI で選択可能

Usage:
    # デフォルト (mythos_tiny, WikiText-103)
    python training/pretrain_laptop.py

    # ステップ数・バッチを指定
    python training/pretrain_laptop.py --steps 5000 --micro_batch 2

    # チェックポイントから再開
    python training/pretrain_laptop.py --resume  # 最新チェックポイントから自動再開

    # モデル variant を変更 (メモリに収まる範囲で)
    python training/pretrain_laptop.py --model 1b  # 要16GB+ RAM
"""

import argparse
import logging
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import BushidoMythos
from bushido_mythos import (
    mythos_tiny, mythos_1b, mythos_3b, mythos_10b,
    mythos_50b, mythos_100b, mythos_500b, mythos_1t,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pretrain")


def _safe_torch_load(path: str, allow_unsafe: bool = False) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_err:
        if not allow_unsafe:
            raise RuntimeError(
                f"{path!r} を weights_only=True でロードできませんでした: {first_err}\n"
                "自分で作成した信頼できる checkpoint であれば allow_unsafe=True を渡してください。"
            ) from first_err
        import warnings
        warnings.warn(
            f"weights_only=True が失敗したため weights_only=False にフォールバックします ({path!r}): {first_err}",
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


VARIANTS = {
    "tiny":  mythos_tiny,
    "1b":    mythos_1b,
    "3b":    mythos_3b,
    "10b":   mythos_10b,
    "50b":   mythos_50b,
    "100b":  mythos_100b,
    "500b":  mythos_500b,
    "1t":    mythos_1t,
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(vocab_size: int):
    """
    MythosTokenizer があればそれを使い、なければ GPT-2 tokenizer にフォールバック。

    Returns:
        tokenizer  -- .encode(str) -> list[int] を持つオブジェクト
        actual_vocab_size -- 実際の語彙サイズ
    """
    try:
        from bushido_mythos.tokenizer import MythosTokenizer
        tok = MythosTokenizer()
        log.info(f"Tokenizer: MythosTokenizer  vocab_size={tok.vocab_size:,}")
        return tok, tok.vocab_size
    except Exception:
        pass

    from transformers import AutoTokenizer

    class _GPT2Tok:
        def __init__(self, max_vocab: int):
            self._tok = AutoTokenizer.from_pretrained("gpt2")
            self.vocab_size = min(self._tok.vocab_size, max_vocab)

        def encode(self, text: str) -> list:
            ids = self._tok.encode(text, add_special_tokens=False)
            return [i for i in ids if i < self.vocab_size]

    tok = _GPT2Tok(vocab_size)
    log.info(f"Tokenizer: GPT-2 (clipped to {tok.vocab_size:,})")
    return tok, tok.vocab_size


# ---------------------------------------------------------------------------
# Dataset — WikiText / FineWeb-Edu streaming
# ---------------------------------------------------------------------------

class TextDataset(IterableDataset):
    """
    HuggingFace datasets をストリーミングで読み込み、固定長 (input, target) ペアを生成する。

    `dataset_name="wikitext"` と `dataset_config="wikitext-103-raw-v1"` がデフォルト。
    `dataset_name="HuggingFaceFW/fineweb-edu"` / `dataset_config="sample-10BT"` で
    FineWeb-Edu に切り替え可能（クラウド推奨）。

    ドキュメントをローリングバッファに連結してチャンク化するので、
    docs が短くてもパディングなしで seq_len を均一に保てる。
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        text_field: str = "text",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        from datasets import load_dataset
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if num_workers > 1:
            ds = ds.shard(num_shards=num_workers, index=worker_id)

        buf: list[int] = []
        for sample in ds:
            text = sample.get(self.text_field, "")
            if not text or not text.strip():
                continue
            buf.extend(self.tokenizer.encode(text))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1:]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:],  dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule — linear warmup → cosine decay  (3b_fine_web_edu.py と同構造)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing — atomic write + keep_last rotation  (3b_fine_web_edu.py 準拠)
# ---------------------------------------------------------------------------

def _list_ckpts(ckpt_dir: str) -> list[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg,
    vocab_size: int,
    ckpt_dir: str,
    keep_last: int = 3,
) -> None:
    """チェックポイントをアトミックに書き込み、古いファイルを削除する。"""
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg.__dict__,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)

    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            log.warning(f"古いチェックポイントの削除失敗: {old}: {exc}")

    log.info(f"Checkpoint → {final_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> int:
    """モデルとオプティマイザの状態を復元し、再開ステップを返す。"""
    ckpt = _safe_torch_load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt["step"])
    log.info(f"Resumed from step {step}  ({path})")
    return step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = get_device()
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model variant
    # ------------------------------------------------------------------
    cfg = VARIANTS[args.model]()
    tokenizer, vocab_size = build_tokenizer(cfg.vocab_size)
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = args.seq_len
    cfg.loop_curriculum = args.loop_curriculum
    cfg.act_aux_loss_weight = args.act_aux_loss_weight

    model = BushidoMythos(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: mythos_{args.model}  params={n_params:,} ({n_params/1e6:.1f}M)")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    grad_accum     = args.grad_accum
    total_steps    = args.steps
    warmup_steps   = args.warmup
    max_lr         = args.lr
    min_lr         = max_lr * 0.1
    log_every      = args.log_every
    ckpt_every     = args.ckpt_every
    ckpt_dir       = args.ckpt_dir
    global_batch_tok = args.micro_batch * grad_accum * args.seq_len

    log.info(
        f"seq_len={args.seq_len} | micro_batch={args.micro_batch} | "
        f"grad_accum={grad_accum} | global_batch_tokens={global_batch_tok:,} | "
        f"total_steps={total_steps:,}"
    )

    # ------------------------------------------------------------------
    # Optimizer  (fused=True は CUDA 専用なので使わない)
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        existing = _list_ckpts(ckpt_dir)
        if existing:
            start_step = load_checkpoint(model, optimizer, existing[-1])
        else:
            log.warning(f"--resume 指定だがチェックポイントが見つかりません: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = TextDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split="train",
        text_field=args.text_field,
    )
    # MPS/CPU: pin_memory=False, num_workers=0 (MPSはfork不可のため)
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # AMP context — MPS は autocast 非対応なので nullcontext
    # ------------------------------------------------------------------
    if device.type == "cuda":
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = start_step

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        ce_loss_accum = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device)
            y = y.to(device)

            with amp_ctx:
                logits = model(x)
                ce_loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = (ce_loss + model._last_aux_loss) / grad_accum

            loss.backward()
            loss_accum += loss.item()
            ce_loss_accum += ce_loss.item() / grad_accum

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.update_moe_router_bias(bias_lr=args.moe_bias_lr)
        step += 1

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = global_batch_tok * log_every / dt
            ppl = math.exp(min(ce_loss_accum, 20))  # CE perplexity only
            log.info(
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} | ce {ce_loss_accum:.4f} | ppl {ppl:.1f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e} "
                f"| {tok_per_sec/1e3:.1f}K tok/s"
            )
            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir)

    # 最終チェックポイント (ckpt_every と total_steps が一致しない場合)
    if step > start_step and step % ckpt_every != 0:
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir)

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BushidoMythos laptop pretraining (MPS/CPU対応)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model",        default="tiny",      choices=list(VARIANTS), help="BushidoMythos variant")

    # Dataset
    p.add_argument("--dataset_name",   default="wikitext",              help="HuggingFace dataset name")
    p.add_argument("--dataset_config", default="wikitext-103-raw-v1",   help="HuggingFace dataset config")
    p.add_argument("--text_field",     default="text",                   help="テキストフィールド名")

    # Sequence / batch
    p.add_argument("--seq_len",      type=int,   default=256,   help="シーケンス長 (≤ max_seq_len)")
    p.add_argument("--micro_batch",  type=int,   default=4,     help="マイクロバッチサイズ")
    p.add_argument("--grad_accum",   type=int,   default=4,     help="勾配蓄積ステップ数 (実効バッチ = micro_batch × grad_accum)")

    # Training
    p.add_argument("--steps",        type=int,   default=5000,  help="総ステップ数")
    p.add_argument("--warmup",       type=int,   default=200,   help="LRウォームアップステップ数")
    p.add_argument("--lr",           type=float, default=3e-4,  help="ピーク学習率")

    # Logging / checkpointing
    p.add_argument("--log_every",    type=int,   default=50,    help="ログ出力間隔 (steps)")
    p.add_argument("--ckpt_every",   type=int,   default=500,   help="チェックポイント保存間隔 (steps)")
    p.add_argument("--ckpt_dir",     default="checkpoints/laptop", help="チェックポイント保存先")
    p.add_argument("--resume",       action="store_true",        help="最新チェックポイントから再開")

    # New feature flags
    p.add_argument("--loop_curriculum",     action="store_true",   help="訓練中にループ数をランダムサンプリング（深さ外挿訓練）")
    p.add_argument("--act_aux_loss_weight", type=float, default=0.0, help="ACT 補助損失の重み（0=無効）")
    p.add_argument("--moe_bias_lr",         type=float, default=1e-3, help="MoE router bias の更新率")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
