#!/usr/bin/env python3
"""
8-bit optimizer 比較実験 — fp32 AdamW vs bitsandbytes AdamW8bit。

同一モデル・同一データ・同一シードで、両 optimizer を N ステップ回し、
  (1) loss 軌跡がほぼ一致するか（8-bit がほぼ無損失か）
  (2) ピーク GPU メモリがどれだけ減るか（オプティマイザ状態 8→2 byte/param）
を比較する。

データは乱数トークン（optimizer 比較に内容は無関係・キャッシュ/ネットワーク非依存）。

注意:
  - 8-bit Adam は CUDA + bitsandbytes が必要。無い環境では make_optimizer が通常
    AdamW にフォールバックするため、両条件とも fp32 になり「差なし」になる
    （= GPU で実行すること）。

使い方:
    python training/exp_optim8bit.py --steps 50 --batch_size 4 --seq_len 256
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.finance_pretrain import make_optimizer
from training.eval_perplexity import load_model


def _load_cfg(ckpt_path, allow_unsafe):
    """checkpoint から cfg だけ読む（モデルを VRAM に載せない）。"""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe:
            raise RuntimeError(
                f"{ckpt_path} を weights_only=True で読めません。"
                "信頼できる checkpoint なら --allow_unsafe_checkpoint を付けてください。")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    from bushido_mythos import MythosConfig
    return MythosConfig(**ck["cfg"])


def _random_batches(n, B, T, vocab, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, vocab, (B, T), generator=g)
        y = torch.randint(0, vocab, (B, T), generator=g)
        out.append((x.to(device), y.to(device)))
    return out


def _train(model, cfg, batches, lr, optim8bit, device):
    opt = make_optimizer(model.parameters(), lr, optim8bit=optim8bit)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    losses, t0 = [], time.time()
    for x, y in batches:
        # n_loops を固定して forward を決定化（loop_curriculum のランダム性を排除し、
        # 差が optimizer のみに由来するようにする）
        logits = model(x, n_loops=cfg.max_loop_iters)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) + model._last_aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else float("nan")
    return losses, peak_mb, elapsed


def main():
    p = argparse.ArgumentParser(description="8-bit optimizer 比較")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase1_final.pt")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type != "cuda":
        print("[note] CUDA 無し → 8-bit は通常 AdamW にフォールバックします（差は出ません）。")

    # cfg だけ軽量読込（モデルを VRAM に残さない → ピークメモリ測定を汚さない）
    cfg0 = _load_cfg(args.ckpt, args.allow_unsafe_checkpoint)
    nparams = None

    batches = _random_batches(args.steps, args.batch_size, args.seq_len, cfg0.vocab_size, device)

    results = {}
    import random as _random
    for tag, o8 in [("fp32 AdamW", False), ("8-bit AdamW", True)]:
        print(f"\n=== {tag} ===")
        torch.manual_seed(0); _random.seed(0)  # forward/サンプリングを両条件で揃える
        model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
        model.train()
        if nparams is None:
            nparams = sum(p.numel() for p in model.parameters())
        losses, peak_mb, elapsed = _train(model, cfg, batches, args.lr, o8, device)
        results[tag] = (losses, peak_mb, elapsed)
        print(f"  loss: {losses[0]:.4f} → {losses[-1]:.4f}  | peak={peak_mb:.0f}MB  | {elapsed:.1f}s")
        del model
        if device.type == "cuda":
            # reserved memory を解放（次 run の peak 測定をきれいにし OOM を避ける。
            # peak のリセット自体は _train の冒頭で行う）
            torch.cuda.empty_cache()

    # ── サマリ ───────────────────────────────────────────────
    (l32, m32, t32) = results["fp32 AdamW"]
    (l8,  m8,  t8)  = results["8-bit AdamW"]
    max_ldiff = max(abs(a - b) for a, b in zip(l32, l8))

    print("\n" + "=" * 64)
    print(f"  8-bit optimizer 比較  (params={nparams/1e6:.1f}M, steps={args.steps})")
    print("=" * 64)
    print(f"  理論オプティマイザ状態: fp32={nparams*8/1e6:.0f}MB  →  8-bit≈{nparams*2/1e6:.0f}MB")
    print(f"  {'':<14}{'final loss':>12}{'peak MB':>10}{'time s':>9}")
    print(f"  {'fp32 AdamW':<14}{l32[-1]:>12.4f}{m32:>10.0f}{t32:>9.1f}")
    print(f"  {'8-bit AdamW':<14}{l8[-1]:>12.4f}{m8:>10.0f}{t8:>9.1f}")
    if device.type == "cuda":
        print(f"  ピークメモリ削減: {m32 - m8:.0f}MB ({(1 - m8/m32)*100:.0f}%)")
    print(f"  loss の最大差(全step): {max_ldiff:.5f}  "
          f"({'ほぼ一致=無損失' if max_ldiff < 0.05 else '要確認'})")
    print("=" * 64)


if __name__ == "__main__":
    main()
