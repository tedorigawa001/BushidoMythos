"""
BushidoMythos: kv_down 層 ターゲット QAT（量子化対応学習）— 実装版
================================================================
目的: フル INT8 で finance PPL を大きく悪化させる単一層
      recurrent.block.attn.kv_down を「INT8 量子化に強い重み」へ学習し直す。

実測の裏付け（phase5_final.pt / finance / 30 chunks / n_loops=8）:
  full-INT8 +45.4%（README +46.6% を再現） / mixed-INT8(kv_down=fp32) +16.8%
  → kv_down 単独で INT8 劣化の約 63% を占有（支配的ボトルネック）。
  → ループ増幅は観測されず（ACT 早期停止）。よって当初案の loop-aware 整合項は
     根拠が無く、本実装では採用しない（CE 損失のみのシンプル QAT）。

設計（データで確定）:
  - kv_down だけ Fake-Quant でラップし、kv_down(weight/bias) のみ学習・他層は凍結。
  - Fake-Quant は評価器（torch.quantization.quantize_dynamic）と同じ量子化に揃える:
      per-tensor 対称・zero_point=0・scale = max|w|/127・STE で round を素通し。
    ※ 当初案の LSQ（学習可能スケール）は不採用。評価時に observer がスケールを
       再計算するため学習したスケールが転送されず、weight 適応に意味が出ないため。
  - quant_strength を 0→1 へ線形に上げる段階的量子化（学習序盤は fp32 寄り）。
  - 仕上げ後は kv_down を素の nn.Linear に畳み戻し（QAT 済み fp32 重みを保存）、
    finance_pretrain 形式（cfg=dict / model_state）で *_qat.pt として保存する。
    → eval_qat_compare.py の条件D（full-INT8 QAT後）が load_model で読めて、
       quantize_dynamic を当てて +16.8% にどこまで迫るかを判定できる。

実行:
  python3 training/experiments/qat_kv_down.py \
    --base_ckpt checkpoints/finance_a100_v2/phase5_final.pt \
    --train_cache finance_domain_mix_gpt2 \
    --steps 500 --n_loops 8 --out checkpoints/finance_a100_v2/phase5_qat.pt
"""

import argparse
import os
import sys

# training/experiments/ から training.* / bushido_mythos.* を import 可能にする。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Fake-Quant（評価器 quantize_dynamic と同一の量子化を学習ループへ）
# ============================================================
def fake_quant_symmetric(w: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    """
    per-tensor 対称・zero_point=0 の INT8 fake-quant。
    scale = max|w|/qmax（重みから算出、autograd 上は定数として detach）。
    STE: round の勾配を 1 として通し、weight には勾配が流れるようにする。
    torch.quantization.quantize_dynamic の weight 量子化に揃えてある。
    """
    scale = (w.detach().abs().max() / qmax).clamp_min(1e-12)
    w_scaled = w / scale
    w_clamped = torch.clamp(w_scaled, -qmax, qmax)
    # STE: forward は round、backward は恒等
    w_q = (torch.round(w_clamped) - w_clamped).detach() + w_clamped
    return w_q * scale


class QATLinearKVDown(nn.Module):
    """
    kv_down(nn.Linear) を fake-quant でラップするドロップイン。
    weight だけ量子化（bias は fp32）。quant_strength で fp32↔量子化をブレンド。
    """
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        # 既存の重み/バイアスをそのまま Parameter として継承（学習対象）
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.quant_strength = 0.0   # 0=fp32相当, 1=フルINT8。段階的に上げる

    def forward(self, x):
        w_fq = fake_quant_symmetric(self.weight)
        qs = self.quant_strength
        w = (1.0 - qs) * self.weight + qs * w_fq
        return F.linear(x, w, self.bias)


def _get_kv_down_parent(model):
    """kv_down を持つ親モジュールと属性名を返す（MLA のみ存在）。"""
    parent = model.recurrent.block.attn
    if not hasattr(parent, "kv_down"):
        raise RuntimeError(
            "kv_down が見つかりません。MLA モデルのみ対象です "
            "(GQA には kv_down がありません)。"
        )
    return parent, "kv_down"


def swap_kv_down_to_qat(model):
    """recurrent.block.attn.kv_down を QAT 版に差し替える。"""
    parent, attr = _get_kv_down_parent(model)
    target = getattr(parent, attr)
    setattr(parent, attr, QATLinearKVDown(target))
    return model


def fold_qat_back_to_linear(model):
    """
    QATLinearKVDown を素の nn.Linear に畳み戻す。
    学習済み(量子化に強い)fp32 weight/bias をそのまま持たせ、保存・評価で
    標準の state_dict キーとして load_model から読めるようにする。
    """
    parent, attr = _get_kv_down_parent(model)
    qat = getattr(parent, attr)
    if not isinstance(qat, QATLinearKVDown):
        return model  # 既に Linear
    out_features, in_features = qat.weight.shape
    lin = nn.Linear(in_features, out_features, bias=qat.bias is not None)
    with torch.no_grad():
        lin.weight.copy_(qat.weight)
        if qat.bias is not None:
            lin.bias.copy_(qat.bias)
    setattr(parent, attr, lin)
    return model


def freeze_all_but_kv_down(model):
    """
    kv_down(weight/bias) 以外を凍結し、学習対象パラメータのリストを返す。
    「kv_down だけ学習し直す」を保証し、モデル全体のドリフトを防ぐ。
    """
    parent, attr = _get_kv_down_parent(model)
    kv = getattr(parent, attr)
    trainable = list(kv.parameters())          # weight (, bias)
    trainable_ids = {id(p) for p in trainable}
    for p in trainable:
        p.requires_grad = True
    for p in model.parameters():
        if id(p) not in trainable_ids:
            p.requires_grad = False
    return trainable


def qat_finetune_phase(model, dataloader, steps=500, n_loops=8, lr=2e-5,
                       log_every=50):
    """
    kv_down ターゲット QAT 仕上げ。CE 損失のみ（loop-aware なし）。
    quant_strength を 0→1 へ線形に上げる（前半でフル量子化へ到達）。
    kv_down(weight/bias) のみ学習し、他は凍結する。

    dataloader は (x, y) タプル（finance_pretrain の dataset と同じ向き）を yield。
    """
    model = swap_kv_down_to_qat(model)
    parent, attr = _get_kv_down_parent(model)
    kv = getattr(parent, attr)

    trainable = freeze_all_but_kv_down(model)
    opt = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    for step, (x, y) in enumerate(dataloader):
        if step >= steps:
            break
        kv.quant_strength = min(1.0, step / max(1.0, steps * 0.5))  # 前半でフル量子化へ
        logits = model(x, n_loops=n_loops)            # [B, T, V]
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:>4}/{steps}  qs={kv.quant_strength:.2f}  "
                  f"loss={loss.item():.4f}")

    kv.quant_strength = 1.0           # 最終的にフル INT8 前提
    model.eval()
    fold_qat_back_to_linear(model)    # 素の nn.Linear に畳み戻して保存可能に
    return model


# ============================================================
# データローダ（finance キャッシュからフラット token を (x,y) で供給・オフライン可）
# ============================================================
def finance_token_loader(ids: torch.Tensor, seq_len: int, batch_size: int,
                         device: torch.device, seed: int = 0):
    """flat token 列から (x, y) バッチを無限に供給。専用 Generator で決定的。"""
    gen = torch.Generator().manual_seed(seed)
    n = ids.numel() - seq_len - 1
    if n <= 0:
        raise ValueError(f"token 数 {ids.numel()} が seq_len={seq_len} に対して不足。")
    while True:
        starts = torch.randperm(n, generator=gen)[:batch_size]
        x = torch.stack([ids[s:s + seq_len] for s in starts]).to(device)
        y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts]).to(device)
        yield x, y


# ============================================================
# CLI
# ============================================================
def build_argparser():
    p = argparse.ArgumentParser(description="kv_down ターゲット QAT 仕上げ")
    p.add_argument("--base_ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt",
                   help="QAT 前のベース（phase5_final.pt）")
    p.add_argument("--train_cache", default="finance_domain_mix_gpt2",
                   help="QAT 学習に使う finance キャッシュ名（.cache/<name>_<vocab>_v1.pt）。"
                        "Phase3 と同じ domain mix が既定。")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--steps", type=int, default=500, help="QAT ステップ数（仕上げなので短め）")
    p.add_argument("--n_loops", type=int, default=8, help="学習時の recurrent ループ数")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   help="既定 cpu。GPU(Colab)なら cuda で高速化")
    p.add_argument("--out", default="checkpoints/finance_a100_v2/phase5_qat.pt")
    return p


def main():
    args = build_argparser().parse_args()
    from training.eval_perplexity import load_model
    from training.exp_quantize_ablation import _load_token_ids

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # 1) ベースをロード（model_state / cfg=dict / compile 接頭辞を堅牢処理）
    model, cfg = load_model(args.base_ckpt, device)

    # 2) finance 学習データ（キャッシュ・オフライン可）
    ids = _load_token_ids(args.cache_dir, args.train_cache, cfg.vocab_size)
    if ids is None:
        raise SystemExit(
            f"学習キャッシュが見つかりません: "
            f"{args.cache_dir}/{args.train_cache}_{cfg.vocab_size}_v1.pt")
    ids = ids.clamp(0, cfg.vocab_size - 1)
    print(f"  train tokens: {ids.numel():,}  (cache={args.train_cache})")
    loader = finance_token_loader(ids, args.seq_len, args.batch_size, device, seed=args.seed)

    # 3) QAT 仕上げ
    print(f"QAT: kv_down ターゲット / steps={args.steps} n_loops={args.n_loops} "
          f"lr={args.lr} bs={args.batch_size} device={args.device}")
    model = qat_finetune_phase(model, loader, steps=args.steps,
                               n_loops=args.n_loops, lr=args.lr)

    # 4) 保存（finance_pretrain 形式: cfg=dict + model_state）。
    #    cfg dict は元 ckpt から取り出して流用（QAT で cfg は不変）。
    raw = torch.load(args.base_ckpt, map_location="cpu", weights_only=True)
    cfg_dict = raw["cfg"]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "cfg": cfg_dict,
        "model_state": model.state_dict(),
        "step": raw.get("step", 0),
        "tag": "kv_down_qat",
    }, args.out)
    print(f"[saved] {args.out}")
    print("評価: eval_qat_compare.py に --qat_ckpt で渡すと条件D(full-INT8 QAT後)が出ます:")
    print(f"  python3 training/experiments/eval_qat_compare.py \\\n"
          f"    --base_ckpt {args.base_ckpt} --qat_ckpt {args.out} \\\n"
          f"    --eval_set finance --n_loops 1,2,4,8 --eval_max_chunks 30")


if __name__ == "__main__":
    main()
