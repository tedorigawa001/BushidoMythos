"""
Laptop-compatible training script for BushidoMythos mythos_tiny (~3.7M params).

Supports MPS (Apple Silicon) with CPU fallback.
No FSDP, no torchrun, no fused AdamW, no bfloat16 autocast.

Usage:
    python training/train_tiny.py
    python training/train_tiny.py --steps 2000 --batch_size 4
    python training/train_tiny.py --dataset wikitext  # needs: pip install datasets
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Allow running from repo root or training/ directory
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos, mythos_tiny

# ──────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────
# Safe checkpoint loading
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────

class SyntheticDataset:
    """Random token sequences — no external dependency."""

    def __init__(self, vocab_size: int, seq_len: int, n_batches: int, batch_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len + 1), device=self.device)
            yield tokens[:, :-1], tokens[:, 1:]


class WikitextDataset:
    """Streams WikiText-2 via HuggingFace datasets (needs: pip install datasets)."""

    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, device: torch.device,
                 cache_dir: str = ".cache"):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        import os
        print("Loading WikiText-2 and tokenizer …")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = Path(cache_dir) / f"wikitext2_gpt2_{vocab_size}.pt"

        if cache_path.exists():
            print(f"  Loading cached tokens from {cache_path}")
            ids = torch.load(cache_path, weights_only=True)
        else:
            print("  Tokenizing row-by-row (first run, will cache) …")
            tok = AutoTokenizer.from_pretrained("gpt2")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            all_ids: list[int] = []
            for row in ds["text"]:
                if row.strip():
                    toks = tok.encode(row, add_special_tokens=False)
                    all_ids.extend(min(t, vocab_size - 1) for t in toks)
            ids = torch.tensor(all_ids, dtype=torch.long)
            torch.save(ids, cache_path)
            print(f"  Cached to {cache_path}")

        self._ids = ids
        print(f"WikiText-2: {len(ids):,} tokens loaded.")

    def __len__(self) -> int:
        return max(1, (len(self._ids) - self.seq_len - 1) // (self.batch_size * self.seq_len))

    def __iter__(self):
        ids = self._ids
        n = len(ids) - self.seq_len - 1
        starts = torch.randperm(n)[: len(self) * self.batch_size]
        for i in range(0, len(starts) - self.batch_size + 1, self.batch_size):
            batch_starts = starts[i : i + self.batch_size]
            x = torch.stack([ids[s : s + self.seq_len] for s in batch_starts]).to(self.device)
            y = torch.stack([ids[s + 1 : s + self.seq_len + 1] for s in batch_starts]).to(self.device)
            yield x, y


# ──────────────────────────────────────────────────────────────
# LR schedule helpers
# ──────────────────────────────────────────────────────────────

def lr_lambda(step: int, warmup: int, total: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}")

    # Model
    cfg: MythosConfig = mythos_tiny()
    overrides = {
        "max_seq_len": args.seq_len,
        "loop_curriculum": args.loop_curriculum,
        "act_aux_loss_weight": args.act_aux_loss_weight,
        "use_hyper_connections": args.use_hyper_connections,
        "use_depth_attn": args.use_depth_attn,
    }
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    cfg = MythosConfig(**{**cfg.__dict__, **overrides})
    model = BushidoMythos(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  ({n_params / 1e6:.1f}M)")

    # Dataset
    if args.dataset == "wikitext":
        try:
            dataset = WikitextDataset(cfg.vocab_size, args.seq_len, args.batch_size, device)
        except Exception as e:
            print(f"WikiText failed ({e}), falling back to synthetic data.")
            dataset = SyntheticDataset(cfg.vocab_size, args.seq_len, args.steps, args.batch_size, device)
    else:
        dataset = SyntheticDataset(cfg.vocab_size, args.seq_len, args.steps, args.batch_size, device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    def _lr(step: int) -> float:
        return lr_lambda(step, args.warmup_steps, args.steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)

    # Checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if specified or auto-detect latest
    step = 0
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        candidates = sorted(ckpt_dir.glob("step_*.pt"))
        if candidates:
            resume_path = str(candidates[-1])
    if resume_path:
        print(f"Resuming from: {resume_path}")
        ckpt = _safe_torch_load(resume_path)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        step = ckpt["step"]
        # Advance scheduler to match resumed step
        for _ in range(step):
            scheduler.step()
        print(f"  Resumed at step {step}")

    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    t0 = time.time()

    print(f"\nStarting training: {args.steps} steps, batch={args.batch_size}, seq_len={args.seq_len}")
    print(f"Logging every {args.log_every} steps, saving every {args.save_every} steps\n")

    epoch = 0
    while step < args.steps:
        epoch += 1
        for x, y in dataset:
            if step >= args.steps:
                break

            # Forward
            logits = model(x)                              # (B, T, vocab)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1),
            )
            loss = ce_loss + model._last_aux_loss

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            model.update_moe_router_bias()
            scheduler.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_aux_loss += model._last_aux_loss.item()
            step += 1

            if step % args.log_every == 0:
                avg_loss = total_loss / args.log_every
                avg_ce = total_ce_loss / args.log_every
                avg_aux = total_aux_loss / args.log_every
                ppl = math.exp(min(avg_ce, 20))  # CE perplexity only
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"step {step:6d}/{args.steps}  "
                    f"loss={avg_loss:.4f}  ce={avg_ce:.4f}  aux={avg_aux:.4f}  ppl={ppl:.1f}  "
                    f"lr={lr_now:.2e}  "
                    f"elapsed={elapsed:.0f}s"
                )
                total_loss = 0.0
                total_ce_loss = 0.0
                total_aux_loss = 0.0
                t0 = time.time()

            if step % args.save_every == 0:
                ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "cfg": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"  → Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = ckpt_dir / "final.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg.__dict__,
        },
        final_path,
    )
    print(f"\nTraining complete. Final checkpoint: {final_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BushidoMythos tiny on a laptop (MPS/CPU).")
    p.add_argument("--steps",        type=int,   default=1000,    help="Total training steps")
    p.add_argument("--batch_size",   type=int,   default=4,       help="Micro-batch size")
    p.add_argument("--seq_len",      type=int,   default=128,     help="Sequence length (≤512 for tiny)")
    p.add_argument("--lr",           type=float, default=3e-4,    help="Peak learning rate")
    p.add_argument("--warmup_steps", type=int,   default=100,     help="LR warmup steps")
    p.add_argument("--grad_clip",    type=float, default=1.0,     help="Gradient clipping norm")
    p.add_argument("--log_every",    type=int,   default=50,      help="Log frequency (steps)")
    p.add_argument("--save_every",   type=int,   default=500,     help="Checkpoint frequency (steps)")
    p.add_argument("--ckpt_dir",     type=str,   default="checkpoints/tiny", help="Checkpoint directory")
    p.add_argument("--dataset",             type=str,   default="synthetic",
                   choices=["synthetic", "wikitext"],
                   help="Dataset: 'synthetic' (no deps) or 'wikitext' (needs datasets+transformers)")
    p.add_argument("--loop_curriculum",     action="store_true", help="Randomise n_loops in [1, max_loop_iters] each step")
    p.add_argument("--act_aux_loss_weight", type=float, default=0.0,  help="ACT ponder-cost loss weight (0=disabled)")
    p.add_argument("--use_hyper_connections", action="store_true",    help="Enable hyper-connections (learned α/β residual)")
    p.add_argument("--use_depth_attn",      action="store_true",      help="Enable depth cross-attention across loop iterations")
    p.add_argument("--vocab_size",          type=int,   default=None, help="Override vocab size (e.g. 50257 for full GPT-2)")
    p.add_argument("--resume",              type=str,   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--auto_resume",         action="store_true",      help="Auto-resume from latest checkpoint in ckpt_dir")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
