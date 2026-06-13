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


def build_config(
    dim: int = 768,
    n_heads: int = 12,
    expert_dim: int = 768,
    max_loop_iters: int = 8,
    attn_type: str = "mla",
    n_kv_heads: int = 4,
) -> MythosConfig:
    return MythosConfig(
        vocab_size=50257,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=1024,
        max_loop_iters=max_loop_iters,
        prelude_layers=1,
        coda_layers=1,
        attn_type=attn_type,
        kv_lora_rank=64,
        q_lora_rank=192,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=28,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=expert_dim,
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


def main(out_path: str, use_gpt2_init: bool = True,
         dim: int = 768, n_heads: int = 12, expert_dim: int = 768,
         max_loop_iters: int = 8, attn_type: str = "mla",
         n_kv_heads: int = 4) -> None:
    cfg = build_config(dim=dim, n_heads=n_heads, expert_dim=expert_dim,
                       max_loop_iters=max_loop_iters, attn_type=attn_type,
                       n_kv_heads=n_kv_heads)

    # GPT-2 small の埋め込み (50257×768) は dim=768 のときだけ流用可能。
    # dim が異なる構成では shape が合わないため自動でスキップする。
    if use_gpt2_init and cfg.dim != 768:
        print(f"[note] dim={cfg.dim} != 768 のため GPT-2 埋め込み初期化はスキップします（ランダム初期化）。")
        use_gpt2_init = False

    model = BushidoMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    print("モデル構成:")
    print(f"  attn_type={cfg.attn_type}  dim={cfg.dim}  n_heads={cfg.n_heads}  n_kv_heads={cfg.n_kv_heads}")
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
            "tag": f"A100-base-{cfg.attn_type}-dim{cfg.dim}-{n_params/1e6:.0f}M"
                   f"-loop{cfg.max_loop_iters}{'-gpt2init' if use_gpt2_init else ''}",
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
    p.add_argument("--dim", type=int, default=768,
                   help="隠れ次元 (default: 768)。768 以外では GPT-2 初期化は自動スキップ")
    p.add_argument("--n_heads", type=int, default=12,
                   help="アテンションヘッド数 (default: 12)。dim を割り切り、かつ "
                        "n_kv_heads でも割り切れる値にすること(例 default 12 は 4 で割れる)")
    p.add_argument("--n_kv_heads", type=int, default=4,
                   help="KV ヘッド数 (default: 4)。n_heads はこの値で割り切れる必要あり。"
                        "GQA の圧縮比を変えるならここを調整")
    p.add_argument("--expert_dim", type=int, default=768,
                   help="MoE エキスパート次元 (default: 768)。比例スケールなら dim と同値")
    p.add_argument("--max_loop_iters", type=int, default=8,
                   help="最大再帰ループ数 (default: 8)。loop curriculum の裾を使うなら裾の最大値(例 12)に")
    p.add_argument("--attn_type", choices=["mla", "gqa"], default="mla",
                   help="アテンション種別 (default: mla)。kv_down 脆弱性の MLA vs GQA 比較用に gqa を選択可")
    args = p.parse_args()
    main(args.out, use_gpt2_init=not args.no_gpt2_init,
         dim=args.dim, n_heads=args.n_heads, expert_dim=args.expert_dim,
         max_loop_iters=args.max_loop_iters, attn_type=args.attn_type,
         n_kv_heads=args.n_kv_heads)
