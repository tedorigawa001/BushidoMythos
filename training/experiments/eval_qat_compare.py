"""
eval_qat_compare.py
===================
QAT 前後の INT8 量子化を「同一プロトコル」で並べて比較する評価スクリプト骨格。

比較する 4 条件:
  (A) fp32                      … 基準
  (B) full-INT8 (QATなし)        … README 実測の +46.6% を再現する対象
  (C) mixed-INT8 (kv_down=fp32)  … README の現状回避策
  (D) full-INT8 (QAT後)          … 本提案。kv_down も INT8 にして QAT で頑健化

出力:
  - WikiText (general) と finance セットそれぞれの PPL と、fp32比の悪化率(%)
  - n_loops スイープ（QAT の効果はループ深度で効いてくるため必須）
  - Markdown レポート

設計方針:
  - README の既存スクリプトに寄せる:
      eval_perplexity.py の PPL 計測ロジック / clamp / dtype 扱い
      exp_quantize.py    の INT8 dynamic quant
      make_mixed_int8.py の mixed-INT8 ロード
      chat.py            のチェックポイント選択・トークナイザ
  - QAT チェックポイントは qat_kv_down.py の qat_finetune_phase で
    作った *_qat.pt を --qat_ckpt で渡す前提。

注意: torch / datasets / 各 README スクリプトに依存。CPU で実行すると
      INT8 dynamic と再現性が安定する（eval_finance_behavior.py と同じ理由）。
"""

import argparse
import os
import sys
import copy

# このファイルは training/experiments/ にあるため、`training.*` / `bushido_mythos.*` を
# どこから実行しても import できるよう、リポジトリルート（2階層上）を sys.path に追加する。
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ------------------------------------------------------------------
# 依存（実環境で有効化）。本ファイルはネット不可環境のため遅延 import。
# ------------------------------------------------------------------
def _lazy_imports():
    import torch
    from bushido_mythos.main import BushidoMythos, MythosConfig
    return torch, BushidoMythos, MythosConfig


# ==================================================================
# 量子化の 4 条件を作る
# ==================================================================
def make_fp32(model):
    """(A) 何もしない基準。"""
    return model


def make_full_int8(model):
    """
    (B) full-INT8: 全 nn.Linear を INT8 dynamic 量子化。
    README の exp_quantize.py と同じ手法（CPU, キャリブレーション無し）。
    """
    import torch
    return torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )


def make_mixed_int8(model):
    """
    (C) mixed-INT8: kv_down 以外を INT8、kv_down は fp32 で温存。
    README の make_mixed_int8.py 相当。

    重要:
      quantize_dynamic({nn.Linear}) は「型」で対象を選ぶため、kv_down も
      nn.Linear である以上、型集合では除外できない（旧実装のバグ）。
      そこで kv_down を「nn.Linear ではない別型ラッパ」に一時退避してから
      量子化し、量子化対象集合に nn.Linear だけを渡す。ラッパは fp32 のまま
      forward するので kv_down は温存される。

    実プロジェクトでは make_mixed_int8.py のロード関数を直接呼ぶのが最も安全。
    本関数はそれが無い場合のフォールバック実装。
    """
    import torch
    import torch.nn as nn

    class _NonQuantLinear(nn.Module):
        """nn.Linear と同じ計算をするが型が違うので quantize_dynamic に拾われない。"""
        def __init__(self, lin: nn.Linear):
            super().__init__()
            self.weight = lin.weight
            self.bias = lin.bias

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight, self.bias)

    # 1) kv_down を退避（親モジュールと属性名を記録して差し替え）
    swapped = []  # (parent_module, attr_name, original_linear)
    for name, module in list(model.named_modules()):
        for attr, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and "kv_down" in f"{name}.{attr}":
                setattr(module, attr, _NonQuantLinear(child))
                swapped.append((module, attr, child))

    if not swapped:
        raise RuntimeError(
            "kv_down 層が見つかりませんでした。層名を確認してください "
            "(README: recurrent.block.attn.kv_down)。"
        )

    # 2) 残りの nn.Linear だけを INT8 量子化
    qmodel = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # 3) （任意）退避した kv_down を元の nn.Linear 表現に戻す
    #    quantize_dynamic は新モデルを返すが in-place 改変もするため、
    #    比較の独立性を保つには fresh_model() から作り直すのが安全。
    return qmodel


def load_qat_full_int8(qat_ckpt_path):
    """
    (D) QAT後 full-INT8: qat_kv_down.py で仕上げた重みをロードし、
    kv_down を含めて INT8 量子化する。QAT 済みなので頑健であることを期待。

    finance_pretrain 形式の checkpoint（model_state / cfg=dict / compile 接頭辞）を
    堅牢に読むため、eval_perplexity.load_model を再利用する。
    """
    import torch
    from training.eval_perplexity import load_model
    model, _cfg = load_model(qat_ckpt_path, torch.device("cpu"))
    return make_full_int8(model)


# ==================================================================
# PPL 計測（eval_perplexity.py のロジックに合わせた骨格）
# ==================================================================
def compute_ppl(model, token_ids, seq_len, n_loops, vocab_size, device="cpu"):
    """
    非重複チャンクで負の対数尤度を平均し exp する素朴な PPL。
    eval_perplexity.py 同様、token を [0, vocab_size-1] に clamp。
    """
    import torch
    model.eval().to(device)
    nlls, count = [], 0
    ids = token_ids.clamp(0, vocab_size - 1)
    n_chunks = ids.size(0) // seq_len
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(device)
            inp, tgt = chunk[:, :-1], chunk[:, 1:]
            logits = model(inp, n_loops=n_loops)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
            nlls.append(loss.item()); count += 1
    mean_nll = sum(nlls) / max(count, 1)
    return math.exp(mean_nll) if count else float("nan")


import math  # compute_ppl が使う


# ==================================================================
# 評価ドライバ
# ==================================================================
def evaluate_all(base_ckpt, qat_ckpt, datasets, n_loops_list, seq_len, device):
    """
    4 条件 × データセット × n_loops を回し、結果 dict を返す。
    datasets: {"wikitext": tensor_ids, "finance": tensor_ids}
    """
    import torch
    from training.eval_perplexity import load_model

    dev = torch.device(device)
    # finance_pretrain 形式（model_state / cfg=dict / compile 接頭辞 / shape フィルタ）を
    # 堅牢に処理する既存ローダを再利用。cfg/vocab を 1 回だけ先読みする。
    _probe, cfg = load_model(base_ckpt, dev)
    vocab = cfg.vocab_size
    del _probe

    def fresh_model():
        # quantize_dynamic は in-place 改変するため、条件ごとに独立インスタンスを作る
        m, _ = load_model(base_ckpt, dev)
        return m

    builders = {
        "A_fp32":       lambda: make_fp32(fresh_model()),
        "B_full_int8":  lambda: make_full_int8(fresh_model()),
        "C_mixed_int8": lambda: make_mixed_int8(fresh_model()),
        "D_qat_int8":   lambda: load_qat_full_int8(qat_ckpt),
    }

    results = {}  # results[cond][dataset][n_loops] = ppl
    for cond, build in builders.items():
        if cond == "D_qat_int8" and not qat_ckpt:
            continue
        model = build()
        results[cond] = {}
        for dname, ids in datasets.items():
            results[cond][dname] = {}
            for nl in n_loops_list:
                ppl = compute_ppl(model, ids, seq_len, nl, vocab, device)
                results[cond][dname][nl] = ppl
    return results


# ==================================================================
# レポート生成
# ==================================================================
COND_LABELS = {
    "A_fp32":       "fp32 (基準)",
    "B_full_int8":  "full-INT8 (QATなし)",
    "C_mixed_int8": "mixed-INT8 (kv_down=fp32)",
    "D_qat_int8":   "full-INT8 (QAT後)",
}


def render_report(results, n_loops_list, out_path):
    lines = ["# QAT 前後 INT8 量子化 比較レポート\n"]
    for dname in next(iter(results.values())).keys():
        lines.append(f"\n## {dname}\n")
        # ヘッダ
        header = "| 条件 | " + " | ".join(f"n_loops={nl}" for nl in n_loops_list) + " | fp32比(代表) |"
        sep = "|" + "---|" * (len(n_loops_list) + 2)
        lines += [header, sep]
        # fp32 基準（代表として最大 n_loops を使う）
        rep_nl = n_loops_list[-1]
        base_ppl = results.get("A_fp32", {}).get(dname, {}).get(rep_nl)
        for cond in ["A_fp32", "B_full_int8", "C_mixed_int8", "D_qat_int8"]:
            if cond not in results:
                continue
            row = results[cond][dname]
            cells = " | ".join(f"{row[nl]:.2f}" for nl in n_loops_list)
            if base_ppl and cond != "A_fp32":
                deg = (row[rep_nl] - base_ppl) / base_ppl * 100
                deg_str = f"+{deg:.1f}%"
            else:
                deg_str = "—"
            lines.append(f"| {COND_LABELS[cond]} | {cells} | {deg_str} |")
    report = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    return report


# ==================================================================
# CLI
# ==================================================================
def build_argparser():
    p = argparse.ArgumentParser(description="QAT 前後の INT8 量子化を比較評価")
    p.add_argument("--base_ckpt", default="checkpoints/finance_a100_v2/phase5_final.pt",
                   help="QAT前のベース（phase5_final.pt）")
    p.add_argument("--qat_ckpt", default="",
                   help="QAT仕上げ済みチェックポイント（qat_finetune_phase の出力）。空なら D を省略")
    p.add_argument("--n_loops", default="1,2,4,8",
                   help="評価する recurrent ループ数（カンマ区切り）。phase5 は max_loop_iters=8 で "
                        "ACT 早期停止するため 8 超は 8 と同値（クランプ）。意味のある範囲 1〜8 を推奨")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--eval_set", choices=["finance", "wikitext"], default="finance",
                   help="評価分布。finance=cache（元発見 +46.6% と同条件）/ "
                        "wikitext=一般言語（DL）。既定 finance")
    p.add_argument("--finance_eval", default="financial_news_gpt2",
                   help="finance 評価キャッシュ名（.cache/<name>_<vocab>_v1.pt）")
    p.add_argument("--cache_dir", default=".cache",
                   help="finance 評価キャッシュのディレクトリ")
    p.add_argument("--split", default="test",
                   help="WikiText-103 の split（eval_set=wikitext 用。既定 test）")
    p.add_argument("--device", default="cpu",
                   help="INT8 dynamic と再現性のため cpu 推奨（eval_finance_behavior.py と同様）")
    p.add_argument("--out", default="training/report/qat_compare_report.md")
    p.add_argument("--eval_max_chunks", type=int, default=30,
                   help="exp_quantize.py に合わせた評価チャンク上限")
    return p


def _load_wikitext_dataset(split, seq_len, eval_max_chunks):
    """
    既存 eval_perplexity.py の WikiText-103 ローダ／検証付き GPT-2 トークナイザを
    再利用して 1-D token 列を作り、評価チャンク数に切り詰めて返す。
    compute_ppl は ids を seq_len 単位の非重複チャンクに割るため、
    トークン数を eval_max_chunks * seq_len に揃えると exp_quantize.py と同じ評価量になる。
    """
    from training.eval_perplexity import load_wikitext103, _build_gpt2_tokenizer

    tok = _build_gpt2_tokenizer()
    ids = load_wikitext103(split, tok, seq_len)         # 1-D long tensor
    if eval_max_chunks and eval_max_chunks > 0:
        cap = eval_max_chunks * seq_len
        if ids.size(0) > cap:
            ids = ids[:cap]
    print(f"  WikiText({split}): {ids.size(0):,} tokens "
          f"-> {ids.size(0) // seq_len} chunks (seq_len={seq_len})")
    return ids


def _peek_vocab(base_ckpt):
    """finance キャッシュ名に必要な vocab_size を checkpoint cfg から軽量に取得。"""
    import torch
    ck = torch.load(base_ckpt, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    return cfg["vocab_size"] if isinstance(cfg, dict) else cfg.vocab_size


def _load_finance_dataset(cache_dir, finance_eval, vocab, seq_len, eval_max_chunks):
    """
    既存 exp_quantize_ablation._load_token_ids で finance 評価キャッシュ
    (.cache/<name>_<vocab>_v1.pt) を読む。元発見 +46.6% と同条件の domain 内評価。
    compute_ppl が seq_len チャンクに割るため eval_max_chunks*seq_len に切り詰める。
    """
    from training.exp_quantize_ablation import _load_token_ids
    cap = eval_max_chunks * seq_len if eval_max_chunks and eval_max_chunks > 0 else 200_000
    ids = _load_token_ids(cache_dir, finance_eval, vocab, cap=cap)
    if ids is None:
        raise SystemExit(
            f"金融評価キャッシュが見つかりません: "
            f"{cache_dir}/{finance_eval}_{vocab}_v1.pt\n"
            "Colab 側で finance データの cache を作成済みか確認してください。"
        )
    print(f"  finance({finance_eval}): {ids.size(0):,} tokens "
          f"-> {ids.size(0) // seq_len} chunks (seq_len={seq_len})")
    return ids


def main():
    args = build_argparser().parse_args()
    n_loops_list = [int(x) for x in args.n_loops.split(",")]

    # --- 評価分布を選択（finance=domain 内 / wikitext=一般言語）---
    if args.eval_set == "finance":
        vocab = _peek_vocab(args.base_ckpt)
        datasets = {
            "finance": _load_finance_dataset(
                args.cache_dir, args.finance_eval, vocab, args.seq_len, args.eval_max_chunks),
        }
    else:
        datasets = {
            "wikitext": _load_wikitext_dataset(args.split, args.seq_len, args.eval_max_chunks),
        }

    results = evaluate_all(args.base_ckpt, args.qat_ckpt, datasets,
                           n_loops_list, args.seq_len, args.device)
    report = render_report(results, n_loops_list, args.out)
    print("\n" + report)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
