#!/usr/bin/env python3
"""
混合精度 INT8 モデルの作成・保存・再ロード(export/load)。

決定版レシピ: nn.Linear を INT8 dynamic 量子化するが、量子化に弱い層
(既定: recurrent.block.attn.kv_down)だけ fp32 に残す。
ablation で「この1層を fp32 に残すとサイズ削減を保ったまま品質を大きく回復」と判明。

このスクリプトは:
  1. fp32 checkpoint をロード
  2. 指定層を除いて INT8 dynamic 量子化
  3. 量子化 state_dict + メタ情報(cfg / fp32維持層 / 量子化対象名)を保存
  4. (--verify) 別プロセス相当で再構築・ロードし、forward が一致するか確認

再ロード時は「同じ量子化対象集合で quantize_dynamic → load_state_dict」で再構築する。
(動的量子化モデルは構造を再現してから state_dict をロードする必要があるため)

使い方:
    python training/make_mixed_int8.py --ckpt checkpoints/finance_a100_v2/phase5_final.pt \
        --out checkpoints/phase5_mixed_int8.pt --verify
"""

import argparse
import io
import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos
from training.eval_perplexity import load_model


def _quant_names(model, keep_fp32_substrings):
    """fp32 維持 substring に該当しない nn.Linear 名の集合を返す(=量子化対象)。"""
    keep = [s for s in keep_fp32_substrings if s]
    names = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and not any(s in n for s in keep):
            names.append(n)
    return set(names)


def _quantize(model, quant_names):
    try:
        from torch.ao.quantization import quantize_dynamic
    except Exception:
        from torch.quantization import quantize_dynamic
    return quantize_dynamic(model, set(quant_names), dtype=torch.qint8)


_FORMAT = "mixed_int8_dynamic_v1"


def _build_mixed_from_payload(payload, device):
    """保存ペイロードから混合精度モデルを再構築してロードする(load 経路)。"""
    fmt = payload.get("format")
    if fmt != _FORMAT:
        raise ValueError(
            f"未知の format: {fmt!r}(期待 {_FORMAT!r})。"
            "別バージョン/別形式の checkpoint の可能性があります。")
    cfg = MythosConfig(**payload["cfg"])
    model = BushidoMythos(cfg).to(device)
    model.eval()
    qmodel = _quantize(model, payload["quant_names"])   # 同じ集合で構造を再現
    qmodel.load_state_dict(payload["quant_state"])      # 量子化 state をロード
    qmodel.eval()
    return qmodel, cfg


def load_mixed_int8(path, device=torch.device("cpu"), trusted=False):
    """混合精度 INT8 モデルをロードする(安全側=既定では読まない)。

    量子化 state には torch.qint8 等が含まれ weights_only=True で読めないため、
    weights_only=False(=pickle・任意コード実行リスク)が必須。自分で作成した
    信頼できる checkpoint のみ trusted=True で明示的に許可すること。

    Returns: (qmodel, cfg)
    """
    if not trusted:
        raise RuntimeError(
            f"{path!r} の読み込みには weights_only=False(pickle)が必要で、任意コード"
            "実行のリスクがあります。自分で作成した信頼できる checkpoint の場合のみ "
            "load_mixed_int8(..., trusted=True) を指定してください。")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _build_mixed_from_payload(payload, device)


def main():
    p = argparse.ArgumentParser(description="混合精度 INT8 の export/load")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt")
    p.add_argument("--out", default="checkpoints/mixed_int8.pt")
    p.add_argument("--keep_fp32", default="recurrent.block.attn.kv_down",
                   help="fp32 に残す層名の substring(カンマ区切り可)。"
                        "部分一致なので、出力される 'keep fp32 layers' を必ず確認すること")
    p.add_argument("--verify", action="store_true",
                   help="保存→再ロードの roundtrip で forward 一致を確認")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu")  # INT8 dynamic は CPU
    keep_subs = [s.strip() for s in args.keep_fp32.split(",") if s.strip()]

    # 1) fp32 ロード
    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    model.eval()
    all_lin = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    quant_names = _quant_names(model, keep_subs)
    kept = sorted(set(all_lin) - quant_names)
    print(f"Linear total={len(all_lin)}  quantize={len(quant_names)}  keep fp32={len(kept)}")
    print(f"  keep fp32 layers: {kept}")
    if not kept:
        print("  [warn] fp32 維持層が 0 件(全 INT8 になります)")

    # 2) 量子化
    qmodel = _quantize(model, quant_names)

    # 3) 保存(量子化 state + メタ)
    payload = {
        "quant_state": qmodel.state_dict(),
        "cfg": cfg.__dict__,
        "keep_fp32": keep_subs,
        "quant_names": sorted(quant_names),
        "format": "mixed_int8_dynamic_v1",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    size_mb = out.stat().st_size / 1e6
    print(f"Saved mixed-INT8: {out}  ({size_mb:.0f} MB)")

    # 4) roundtrip 検証
    if args.verify:
        print("\n[verify] 再ロードして forward 一致を確認 …")
        qmodel2, _ = load_mixed_int8(out, device, trusted=True)  # 自作=trusted
        torch.manual_seed(0)
        x = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            y1 = qmodel(x, n_loops=cfg.max_loop_iters)
            y2 = qmodel2(x, n_loops=cfg.max_loop_iters)
        max_diff = (y1 - y2).abs().max().item()
        ok = max_diff < 1e-5
        print(f"  forward 最大差: {max_diff:.2e}  → {'OK(一致)' if ok else 'NG(不一致)'}")
        if not ok:
            sys.exit(1)
    print("\nロード方法: load_mixed_int8(out, device, trusted=True)  # 自作の信頼できる ckpt のみ")


if __name__ == "__main__":
    main()
