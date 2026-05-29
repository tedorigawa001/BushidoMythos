#!/usr/bin/env python3
"""
A100 最適化の起点チェックポイントを作成するスクリプト (v2: dim=768, ~99M params)。

構成:
  dim=768, n_heads=12, max_loop_iters=8, seq_len=1024
  n_experts=28, expert_dim=768, attn_type=mla
  loop_curriculum=True（ランダム深度学習で汎化性能向上）
  act_aux_loss_weight=0.001（ACT ウォームアップ: 学習初期の ponder cost 圧を最小化）
  embed.weight を GPT-2 small (wte) で初期化（語彙・次元が一致）

使い方:
  python training/make_base_ckpt.py
  python training/make_base_ckpt.py --out checkpoints/my_base/final.pt
  python training/make_base_ckpt.py --no_gpt2_init  # GPT-2 初期化をスキップ
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos


_DEFAULT_OUT = "checkpoints/a100_v2_gpt2vocab/final.pt"


def build_config() -> MythosConfig:
    return MythosConfig(
        vocab_size=50257,
        dim=768,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=1024,
        max_loop_iters=8,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=64,
        q_lora_rank=192,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=28,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=768,
        act_threshold=0.99,
        act_aux_loss_weight=0.001,
        rope_theta=10000.0,
        lora_rank=8,
        loop_curriculum=True,
    )


def init_from_gpt2(model: BushidoMythos) -> None:
    """GPT-2 small の wte (50257×768) で embed.weight を初期化する。"""
    print("GPT-2 small の埋め込みをロード中...")
    try:
        from transformers import GPT2Model
    except ImportError:
        raise RuntimeError(
            "transformers が必要です: pip install transformers"
        )

    try:
        gpt2 = GPT2Model.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        print("  ローカルキャッシュなし → ネットワークからダウンロードします")
        gpt2 = GPT2Model.from_pretrained("gpt2")

    src = gpt2.wte.weight.data  # [50257, 768]
    dst = model.embed.weight.data  # [50257, 768]
    assert src.shape == dst.shape, (
        f"shape mismatch: GPT-2 wte {src.shape} vs model embed {dst.shape}"
    )
    dst.copy_(src)
    del gpt2
    print("  GPT-2 埋め込み初期化完了 (shape: {})".format(src.shape))


def main(out_path: str, use_gpt2_init: bool = True) -> None:
    cfg = build_config()
    model = BushidoMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print("モデル構成:")
    print(f"  dim={cfg.dim}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}")
    print(f"  max_seq_len={cfg.max_seq_len}  max_loop_iters={cfg.max_loop_iters}")
    print(f"  n_experts={cfg.n_experts}  expert_dim={cfg.expert_dim}")
    print(f"  loop_curriculum={cfg.loop_curriculum}")
    print(f"  act_aux_loss_weight={cfg.act_aux_loss_weight}  (ACT warm-start; scale up to 0.01 after Phase 1)")
    print(f"  パラメータ数: {n_params / 1e6:.1f}M")

    if use_gpt2_init:
        init_from_gpt2(model)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "step": 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "cfg": cfg.__dict__,
            "tag": "A100-base-v2-dim768-99M-gpt2init",
            "phase1_steps": 0,
            "phase2_steps": 0,
            "phase3_steps": 0,
            "phase4_steps": 0,
        },
        out,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"\n保存完了: {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A100最適化の起点チェックポイントを作成 (v2: dim=768)")
    p.add_argument("--out", default=_DEFAULT_OUT,
                   help=f"出力パス (デフォルト: {_DEFAULT_OUT})")
    p.add_argument("--no_gpt2_init", action="store_true",
                   help="GPT-2 埋め込み初期化をスキップ（ランダム初期化）")
    args = p.parse_args()
    main(args.out, use_gpt2_init=not args.no_gpt2_init)
