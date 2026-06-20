"""
BushidoMythos: kv_down 層 ターゲットQAT（量子化対応学習）
=========================================================
目的: フルINT8で finance PPL +46.6% を引き起こす単一層
      recurrent.block.attn.kv_down を「量子化に強い重み」へ学習し直す。

設計（4人の議論より）:
  C: STE + ループ誤差を損失に含める（誤差増幅をモデルに学習させる）
  A: kv_down 層だけ Fake-Quant 挿入 / LSQ 学習可能スケール / 段階的QAT
  D: ボトルネックだけ冗長化（全層均等量子化より理にかなう）

注意: 概念実証の簡略疑似コード。実スケールでの効果は要検証。
"""

import math

# ============================================================
# PART 1: 疑似コード（PyTorch想定 / 実コードの骨格）
# ============================================================
PSEUDOCODE = r'''
import torch
import torch.nn as nn

class LSQFakeQuantize(nn.Module):
    """
    INT8 fake-quant with Learned Step Size (LSQ).
    順伝播: 量子化 -> 逆量子化（学習中も「INT8で動く」体験をさせる）
    逆伝播: STE で round を素通し、スケール s は勾配で学習。
    """
    def __init__(self, init_scale):
        super().__init__()
        # スケールは学習可能パラメータ（固定 max|W| ではない）
        self.log_s = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        self.qmin, self.qmax = -127, 127

    def forward(self, w):
        s = self.log_s.exp()
        w_scaled = w / s
        # --- STE: round の勾配を 1 として通す ---
        w_clamped = torch.clamp(w_scaled, self.qmin, self.qmax)
        w_q = (torch.round(w_clamped) - w_clamped).detach() + w_clamped
        return w_q * s


class QATLinearKVDown(nn.Module):
    """
    kv_down 層を fake-quant でラップ。weight だけ量子化（bias は fp32）。
    既存の nn.Linear を置き換えるドロップイン。
    """
    def __init__(self, base_linear: nn.Linear):
        super().__init__()
        self.weight = base_linear.weight          # 既存重みを継承
        self.bias = base_linear.bias
        init_scale = base_linear.weight.abs().max().item() / 127.0
        self.fq = LSQFakeQuantize(init_scale)
        self.quant_strength = 0.0                 # 0=fp32相当, 1=フルINT8。段階的に上げる

    def forward(self, x):
        w_q = self.fq(self.weight)
        # 段階的ブレンド: 学習序盤は fp32 寄り、徐々に量子化重みへ
        w = (1 - self.quant_strength) * self.weight + self.quant_strength * w_q
        return torch.nn.functional.linear(x, w, self.bias)


def swap_kv_down_to_qat(model):
    """recurrent.block.attn.kv_down だけを QAT 版に差し替える。"""
    target = model.recurrent.block.attn.kv_down       # 犯人の単一層
    model.recurrent.block.attn.kv_down = QATLinearKVDown(target)
    return model


def loop_aware_qat_loss(model, x, y, base_loss_fn, n_loops, lam=0.5):
    """
    C の提案: ループ誤差増幅を損失に含める。
    通常損失 + 「fp32経路と量子化経路の出力差」をループ深度で測ったペナルティ。
    深いループでの量子化ズレを学習が直接見るようにする。

    入力は finance_pretrain.py の dataset に合わせ (x, y) タプルを想定。
    出力 logits は [B, T, V]、ターゲット y は [B, T] のため reshape して CE に渡す。
    """
    out_q = model(x, n_loops=n_loops)                 # [B, T, V]
    B, T, V = out_q.shape
    base = base_loss_fn(out_q.reshape(B * T, V), y.reshape(B * T))

    # 同じ重みで「量子化を切った」参照経路（quant_strength=0 相当）
    kv = model.recurrent.block.attn.kv_down
    saved = kv.quant_strength
    kv.quant_strength = 0.0
    with torch.no_grad():
        out_ref = model(x, n_loops=n_loops)
    kv.quant_strength = saved

    # 量子化経路が参照経路から離れすぎないよう正則化（誤差増幅の抑制）
    consistency = torch.nn.functional.mse_loss(out_q, out_ref)
    return base + lam * consistency


def freeze_all_but_kv_down(model):
    """
    kv_down（QATLinearKVDown の weight/bias）と LSQ スケール log_s 以外を凍結。
    docstring 通り「kv_down だけ学習し直す」を保証し、モデル全体のドリフトを防ぐ。
    返り値: optimizer に渡すべき学習対象パラメータのリスト。
    """
    kv = model.recurrent.block.attn.kv_down
    trainable = set()
    for p in kv.parameters():        # weight, bias, fq.log_s
        p.requires_grad = True
        trainable.add(p)
    for p in model.parameters():
        if p not in trainable:
            p.requires_grad = False
    return [p for p in trainable]


def qat_finetune_phase(model, dataloader, steps=2000, n_loops=8):
    """
    A の提案: Phase 5 の後に短い QAT 仕上げフェーズ。
    quant_strength を 0 -> 1 へ線形に上げる（段階的量子化）。
    kv_down(+ LSQ log_s) のみを学習し、他は凍結する。
    """
    model = swap_kv_down_to_qat(model)
    kv = model.recurrent.block.attn.kv_down

    # --- kv_down 以外を凍結し、学習対象だけ optimizer へ渡す ---
    trainable_params = freeze_all_but_kv_down(model)
    opt = torch.optim.AdamW(trainable_params, lr=2e-5)  # 仕上げなので小さめLR
    base_loss_fn = nn.functional.cross_entropy

    for step, (x, y) in enumerate(dataloader):          # dataset は (x, y) を yield
        if step >= steps:
            break
        kv.quant_strength = min(1.0, step / (steps * 0.5))  # 前半でフル量子化へ
        loss = loop_aware_qat_loss(model, x, y, base_loss_fn, n_loops)
        opt.zero_grad(); loss.backward(); opt.step()

    kv.quant_strength = 1.0   # 最終的にフルINT8前提に固定
    return model
'''

print(PSEUDOCODE)


# ============================================================
# PART 2: 効果シミュレーション（torch非依存・純Python）
# ============================================================
# ループ誤差増幅のモデル化:
#   各ループで kv_down の量子化誤差 eps が注入され、
#   LTIシステム h_{t+1}=A h_t + ... により sum_t rho^t * eps として蓄積する。
#   QAT は (1) 単発の量子化誤差 eps を下げ、
#          (2) loop_aware 損失により実効 rho を下げる（誤差を増幅しにくい重みへ）。

def simulate_finance_ppl_degradation(
    n_loops, eps_per_loop, rho, base_ppl=20.0, sensitivity=0.85
):
    """
    量子化誤差の蓄積量から finance PPL の悪化率(%)を概算する簡易モデル。
    accumulated_error = eps * sum_{t=0}^{T-1} rho^t
    PPL悪化率 = sensitivity * accumulated_error
    """
    if abs(rho - 1.0) < 1e-9:
        geom = n_loops
    else:
        geom = (1 - rho ** n_loops) / (1 - rho)
    accumulated = eps_per_loop * geom
    degradation_pct = sensitivity * accumulated * 100
    degraded_ppl = base_ppl * (1 + degradation_pct / 100)
    return degradation_pct, degraded_ppl, accumulated


if __name__ == "__main__":
    print("=" * 64)
    print("効果シミュレーション: kv_down 量子化誤差のループ蓄積")
    print("=" * 64)
    BASE_PPL = 20.0
    N_LOOPS = 8   # README の chat.py / eval デフォルト

    # --- シナリオ定義 ---
    # full_int8 : QATなし。単発誤差大、rho 高め（誤差増幅しやすい重み）
    #            -> README 実測の +46.6% に合わせて較正
    # qat_eps   : QATで単発の量子化誤差だけ低減（rho は据え置き）
    # qat_full  : QAT + loop_aware損失で rho も低減（誤差を増幅しにくい重み）
    scenarios = {
        "フルINT8（QATなし・現状）":      dict(eps_per_loop=0.0146, rho=0.93),
        "QAT: 単発誤差のみ低減":          dict(eps_per_loop=0.0061, rho=0.93),
        "QAT + loop-aware（誤差増幅抑制）": dict(eps_per_loop=0.0061, rho=0.78),
    }

    print(f"基準 PPL (fp32) = {BASE_PPL}, n_loops = {N_LOOPS}\n")
    print(f"{'シナリオ':<32}{'PPL悪化率':>12}{'悪化後PPL':>12}")
    print("-" * 64)
    results = {}
    for name, p in scenarios.items():
        deg, dppl, acc = simulate_finance_ppl_degradation(
            N_LOOPS, p["eps_per_loop"], p["rho"], BASE_PPL)
        results[name] = deg
        print(f"{name:<32}{deg:>10.1f}%{dppl:>12.2f}")

    print("-" * 64)
    base_deg = list(results.values())[0]
    print("\n[改善率: フルINT8 を基準とした PPL 悪化の削減]")
    for name, deg in results.items():
        if deg == base_deg:
            continue
        recovered = (base_deg - deg) / base_deg * 100
        print(f"  {name:<32} 悪化を {recovered:>5.1f}% 削減")

    # --- ループ深度を変えたときの感度 ---
    print("\n" + "=" * 64)
    print("ループ深度別: 誤差増幅の効き方（rho が効いてくる）")
    print("=" * 64)
    print(f"{'n_loops':>8}{'フルINT8':>14}{'QAT+loop-aware':>18}")
    print("-" * 44)
    for T in [4, 8, 12, 16, 24]:
        d_full, _, _ = simulate_finance_ppl_degradation(T, 0.0146, 0.93, BASE_PPL)
        d_qat,  _, _ = simulate_finance_ppl_degradation(T, 0.0061, 0.78, BASE_PPL)
        print(f"{T:>8}{d_full:>12.1f}%{d_qat:>16.1f}%")

    print("""
注意:
- これは誤差蓄積を幾何級数で近似した概念モデルです。実PPLは
  exp_quantize.py / exp_mixed_precision.py で実測してください。
- eps と rho は「フルINT8 = +46.6%」に較正した仮値。QAT後の値は
  LSQ・loop-aware損失で 'これくらい下がりうる' という目標感です。
- 本質的な主張は2つ:
  (1) QAT は単発の量子化誤差 eps を下げる
  (2) loop-aware 損失は実効 rho を下げ、深いループでの増幅を抑える
  -> 深く考えさせる（n_loops大）ほど (2) の価値が大きい。
""")
