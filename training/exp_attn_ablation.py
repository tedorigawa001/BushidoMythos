#!/usr/bin/env python3
"""
attention 内 sub-projection / 場所別の量子化 ablation。

module 別 ablation で「attention を fp32 に残すと finance PPL が大回復(+46.6%→+14.1%)」
と判明した。本スクリプトは attention の**どこ**が主犯かをさらに分解する:

  sub-projection 別: wo(out_proj) / q系(q_down,q_up_nope,q_up_rope) / kv系(kv_down,kv_up)
                     / RoPE projection(q_up_rope)
  場所別:           recurrent(ループ内・8回実行)vs prelude/coda(1回)

仮説: MLA + recurrent loop により、ループ内 attention の小さな量子化誤差が増幅している。
       → recurrent-attn だけ fp32 に残して大回復するなら、ループ増幅が支持される。

各構成は「INT8 except G(= G を fp32 に残し、他は全 INT8)」。回復が大きい G が主犯。

使い方:
    python training/exp_attn_ablation.py --ckpt checkpoints/finance_a100_v2/phase5_final.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from training.eval_perplexity import (
    load_model, load_wikitext103, _build_gpt2_tokenizer,
)
from training.exp_quantize_ablation import (
    _state_dict_mb, _quantize_names, _eval_ppl, _load_token_ids,
)


def _load_eval_ids(args, cfg):
    """評価トークン列を返す。finance=cache, wikitext=in-distribution(DL)。

    Phase1(WikiText)で同一学習した MLA/GQA を比べる場合は wikitext(学習分布内)
    の方が相対劣化の信号がクリーン。finance は元発見と同条件(domain 内)。
    """
    if args.eval_set == "wikitext":
        tok = _build_gpt2_tokenizer()
        return load_wikitext103("test", tok, seq_len=1024)
    ids = _load_token_ids(args.cache_dir, args.finance_eval, cfg.vocab_size)
    if ids is None:
        print(f"[error] 金融評価キャッシュが無い: {args.finance_eval}"); sys.exit(1)
    return ids

# attention 内 sub-group(名前で判定)。MLA(q_down/q_up/kv_down/kv_up)と
# GQA(wq/wk/wv/wo)の両命名に対応。
# 注: "attn.q_up_rope (RoPE)" は "attn.q (q-side)" の部分集合(重複あり)。
SUBGROUPS = {
    "attn.wo (out_proj)":        lambda n: ".attn.wo" in n,
    "attn.q (q-side)":           lambda n: (".attn.q_down" in n or ".attn.q_up" in n
                                            or ".attn.wq" in n),
    "attn.kv (kv-side)":         lambda n: (".attn.kv_down" in n or ".attn.kv_up" in n
                                            or ".attn.wk" in n or ".attn.wv" in n),
    # --- H1(MLA圧縮固有)vs H2(KV経路一般)の切り分け用に KV 経路を分離 ---
    # MLA: kv_down=低ランク圧縮(主犯候補) / kv_up=展開。GQA: wk/wv=圧縮なしの直接射影。
    # 該当しないアーキでは 0-match → 全INT8と同じ(warn が出る)。
    #
    # 注: 下の substring 版は prelude+recurrent+coda の**全 attention stage**に match する
    # (= 3層分)。元発見の主犯は recurrent.block.attn.kv_down の**1層**なので、
    # 「ループ内 kv_down が主犯」か「全 stage の kv_down 系が効く」かを切り分けるため、
    # recurrent 限定の**完全一致**版も別に用意する。
    "attn.kv_down (all stages)":   lambda n: ".attn.kv_down" in n,
    "attn.kv_up (all stages)":     lambda n: ".attn.kv_up" in n,
    "attn.wk (all stages)":        lambda n: ".attn.wk" in n,
    "attn.wv (all stages)":        lambda n: ".attn.wv" in n,
    # recurrent ループ内の単層のみ(元発見の主犯 = この1層)
    "recurrent kv_down (MLA,1層)": lambda n: n == "recurrent.block.attn.kv_down",
    "recurrent kv_up (MLA,1層)":   lambda n: n == "recurrent.block.attn.kv_up",
    "recurrent wk (GQA,1層)":      lambda n: n == "recurrent.block.attn.wk",
    "recurrent wv (GQA,1層)":      lambda n: n == "recurrent.block.attn.wv",
    "attn.q_up_rope (RoPE,⊂q)":  lambda n: ".attn.q_up_rope" in n,
    "recurrent attn (loop)":     lambda n: n.startswith("recurrent.block.attn."),
    "prelude+coda attn":         lambda n: (".attn." in n) and (not n.startswith("recurrent.")),
    "ALL attn (reference)":      lambda n: ".attn." in n,
}


def main():
    p = argparse.ArgumentParser(description="attention 内 量子化 ablation")
    p.add_argument("--ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt")
    p.add_argument("--eval_max_chunks", type=int, default=30)
    p.add_argument("--finance_eval", default="financial_news_gpt2")
    p.add_argument("--eval_set", choices=["finance", "wikitext"], default="finance",
                   help="評価分布。finance=cache(元発見と同条件)/ "
                        "wikitext=in-distribution(Phase1学習の MLA vs GQA 比較向け)")
    p.add_argument("--cache_dir", default=".cache")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true")
    args = p.parse_args()
    if args.eval_max_chunks <= 0:
        p.error(f"--eval_max_chunks must be > 0 (got {args.eval_max_chunks})")

    device = torch.device("cpu")
    print("Device: cpu (INT8 dynamic is CPU-only)")
    model, cfg = load_model(args.ckpt, device, allow_unsafe=args.allow_unsafe_checkpoint)
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    all_set = set(linear_names)

    print(f"Eval set: {args.eval_set}")
    fin_ids = _load_eval_ids(args, cfg)

    configs = [("fp32 (baseline)", set()), ("INT8 (all)", all_set)]
    for g, pred in SUBGROUPS.items():
        keep = {n for n in linear_names if pred(n)}
        if not keep:
            # 0 件 → このアーキ(命名)に該当 module が無い。結果は全INT8と同じになるので注意。
            print(f"  [warn] subgroup '{g}' matched 0 Linear modules "
                  "(このアーキでは該当なし; 結果は INT8 全部と同じ)")
        configs.append((f"INT8 except {g}", all_set - keep, len(keep)))

    results = []
    for cfg_t in configs:
        label, qnames = cfg_t[0], cfg_t[1]
        nkeep = cfg_t[2] if len(cfg_t) > 2 else (len(linear_names) - len(qnames))
        qmodel = _quantize_names(model, qnames)
        sz = _state_dict_mb(qmodel)
        fin = _eval_ppl(qmodel, cfg, fin_ids, device, args.eval_max_chunks)
        results.append((label, nkeep, sz, fin))
        print(f"  {label:<30} keep_fp32={nkeep:>3}  size={sz:>4.0f}MB  {args.eval_set} PPL={fin:.2f}")
        del qmodel

    ppl_col = f"{args.eval_set} PPL"
    fp32_fin = results[0][3]
    int8_fin = results[1][3]
    print("\n" + "=" * 74)
    print(f"  attention 内 ablation — {ppl_col}(低いほど良い)")
    print("=" * 74)
    print(f"  {'config':<30}{'keep':>5}{'size MB':>9}{ppl_col:>13}{'Δ vs fp32':>11}")
    for label, nk, sz, fin in results:
        d = (fin - fp32_fin) / fp32_fin * 100
        print(f"  {label:<30}{nk:>5}{sz:>9.0f}{fin:>13.2f}{d:>10.1f}%")
    print("=" * 74)
    print(f"  全INT8: {fp32_fin:.1f} → {int8_fin:.1f} ({(int8_fin-fp32_fin)/fp32_fin*100:+.1f}%)")
    print("  回復が大きい G ほど主犯。recurrent-attn が ALL-attn に迫れば『ループ増幅』を支持。")
    print("  注: 'q_up_rope (RoPE)' は 'attn.q (q-side)' の部分集合(重複)。")
    print("  注: 部分評価(max_chunks)。finance 評価は Phase3+ で学習分布と重なりうる。")
    print("  MLA/GQA 比較は recurrent 1層版で判定: MLA 'recurrent kv_down' vs "
          "GQA 'recurrent wk'/'wv' の回復幅(元主犯はこの1層)。")
    print("  'all stages' 版は prelude+recurrent+coda の3層集約なので参考値。")


if __name__ == "__main__":
    main()
