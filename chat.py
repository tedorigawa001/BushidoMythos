#!/usr/bin/env python3
"""
BushidoMythos チャット推論スクリプト。

使い方:
    python chat.py --finance_mode                       # 推奨: Phase 3-5 の SFT 形式に合わせた指示応答モード
    python chat.py --finance_mode --ckpt checkpoints/finance_a100_v2/phase5_final.pt
    python chat.py --finance_mode --temp 0.6 --top_k 40 --loops 8
    python chat.py                                      # 生テキスト補完モード（Phase 3 以前のモデル向け）
    python chat.py --ckpt_dir checkpoints/finance_a100_v2  # ディレクトリ指定で最新を自動選択
    python chat.py --tokenizer gpt2                     # GPT-2 tokenizer を明示指定
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from bushido_mythos import MythosConfig, BushidoMythos


# ---------------------------------------------------------------------------
# デバイス
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# チェックポイント検索
# ---------------------------------------------------------------------------

_PREFERRED_NAMES = [
    "phase5_final.pt", "phase4_final.pt", "phase3_final.pt", "final.pt",
    "phase2_final.pt", "phase1_final.pt",
]

def find_latest_ckpt(ckpt_dir: str) -> Optional[str]:
    """
    ckpt_dir から最適なチェックポイントを選ぶ。
    優先順位: phase5_final.pt > phase4_final.pt > phase3_final.pt > final.pt > phase2_final.pt > phase1_final.pt > 最新 step_*.pt
    """
    base = Path(ckpt_dir)
    if not base.exists():
        return None
    for name in _PREFERRED_NAMES:
        p = base / name
        if p.exists():
            return str(p)
    candidates = sorted(p for p in base.glob("step_*.pt") if not str(p).endswith(".tmp"))
    return str(candidates[-1]) if candidates else None


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

def _strip_compile_prefix(sd: dict) -> dict:
    """torch.compile が付与する _orig_mod. プレフィックスを除去する。"""
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def _safe_torch_load(path: str, allow_unsafe: bool = False) -> dict:
    """weights_only=True でロードを試み、失敗時は allow_unsafe=True の場合のみ fallback する。

    PyTorch checkpoint は pickle ベースのため、悪意ある .pt を weights_only=False で読むと
    任意コード実行のリスクがある。自分で作成した trusted checkpoint のみ allow_unsafe=True を使うこと。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as first_err:
        if not allow_unsafe:
            raise RuntimeError(
                f"{path!r} を weights_only=True でロードできませんでした: {first_err}\n"
                "自分で作成した信頼できる checkpoint であれば --allow_unsafe_checkpoint を付けて再実行してください。"
            ) from first_err
        import warnings
        warnings.warn(
            f"weights_only=True が失敗したため weights_only=False にフォールバックします ({path!r}): {first_err}\n"
            "信頼できる checkpoint のみこの方法でロードしてください。",
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def load_model(ckpt_path: str, device: torch.device, allow_unsafe: bool = False):
    print(f"Loading: {ckpt_path}")
    ckpt = _safe_torch_load(ckpt_path, allow_unsafe=allow_unsafe)

    cfg = MythosConfig(**ckpt["cfg"])
    model = BushidoMythos(cfg).to(device)
    state_key = "model_state" if "model_state" in ckpt else "model"
    sd = _strip_compile_prefix(ckpt[state_key])

    # shape が一致するキーのみロード（RoPE buffer など shape mismatch を回避）
    model_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    skipped = [k for k in sd if k not in filtered]
    if skipped:
        print(f"  [警告] shape 不一致でスキップ ({len(skipped)}): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  [警告] ロードされなかったキー ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  dim={cfg.dim}  vocab={cfg.vocab_size:,}  params={n_params/1e6:.1f}M  step={ckpt['step']}")
    return model, cfg


# ---------------------------------------------------------------------------
# トークナイザ
# ---------------------------------------------------------------------------

def build_tokenizer(vocab_size: int, mode: str = "auto"):
    """
    mode:
      "auto"  — vocab_size==50257 なら gpt2、それ以外は mythos を試みて失敗なら gpt2
      "gpt2"  — 常に GPT-2 tokenizer（HuggingFace retry なし）
      "mythos"— MythosTokenizer を強制（vocab mismatch 注意）
    """
    if mode == "mythos":
        from bushido_mythos.tokenizer import MythosTokenizer
        return MythosTokenizer()

    if mode == "gpt2" or vocab_size == 50257:
        return _build_gpt2_tokenizer(vocab_size)

    # auto: vocab が GPT-2 でない場合のみ MythosTokenizer を試みる
    try:
        from bushido_mythos.tokenizer import MythosTokenizer
        tok = MythosTokenizer()
        if tok.vocab_size == vocab_size:
            return tok
    except Exception:
        pass
    return _build_gpt2_tokenizer(vocab_size)


def _build_gpt2_tokenizer(vocab_size: int):
    from transformers import AutoTokenizer, GPT2TokenizerFast

    def _load_verified_tokenizer():
        attempts = [
            lambda: AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True),
            lambda: GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True),
            lambda: AutoTokenizer.from_pretrained("gpt2", use_fast=True),
            lambda: GPT2TokenizerFast.from_pretrained("gpt2"),
        ]
        errors = []
        for load in attempts:
            try:
                tok = load()
                if tok.encode("Hello", add_special_tokens=False):
                    return tok
                errors.append("loaded tokenizer returned empty ids for 'Hello'")
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        raise RuntimeError(
            "GPT-2 tokenizer load failed or returned empty token ids.\n"
            "Clear the HuggingFace cache or rerun with network access.\n"
            + "\n".join(errors[-3:])
        )

    class _GPT2Tok:
        def __init__(self, vs: int):
            self._t = _load_verified_tokenizer()
            self.vocab_size = min(self._t.vocab_size, vs)

        def encode(self, text: str) -> list[int]:
            raw = self._t.encode(text, add_special_tokens=False)
            if not raw and text:
                raw = self._t(text, add_special_tokens=False).get("input_ids", [])
            return [min(int(i), self.vocab_size - 1) for i in raw]

        def decode(self, ids: list[int]) -> str:
            return self._t.decode(ids, skip_special_tokens=True)

    return _GPT2Tok(vocab_size)



# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

_FINANCE_DISCLAIMER = (
    "⚠ 金融モード: 出力は参考情報であり、投資助言ではありません。"
    " 実際の取引判断の前に公式情報源を必ず確認してください。"
)

# Instruction format used in Phase 3/4 training
_INSTRUCT_PREFIX = "### Instruction:\n"
_INSTRUCT_RESPONSE = "\n\n### Response:\n"
_INSTRUCT_STOP = "\n### "  # truncate at next instruction boundary

# Appended to the instruction so the model includes risk context in its response
_FINANCE_RISK_SUFFIX = (
    "\n\nPlease acknowledge uncertainty where relevant, include risk considerations, "
    "and note that outputs should be verified from authoritative sources before any trading action."
)


def generate(
    model: BushidoMythos,
    cfg: MythosConfig,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    n_loops: int,
    device: torch.device,
    finance_mode: bool = False,
    repetition_penalty: float = 1.3,
) -> str:
    if finance_mode:
        prompt = _INSTRUCT_PREFIX + prompt + _FINANCE_RISK_SUFFIX + _INSTRUCT_RESPONSE

    ids = tokenizer.encode(prompt)
    if not ids:
        print(f"  [警告] tokenizer.encode() が空リストを返しました。トークン0 でフォールバックします。")
        ids = [0]

    # プロンプトが長すぎる場合、左側を切り詰めて最新コンテキストを優先する
    max_prompt_len = cfg.max_seq_len - max_new_tokens
    if max_prompt_len <= 0:
        print(f"  [警告] max_new_tokens={max_new_tokens} が max_seq_len={cfg.max_seq_len} 以上です。"
              f" max_new_tokens を {cfg.max_seq_len // 2} に調整します。")
        max_new_tokens = cfg.max_seq_len // 2
        max_prompt_len = cfg.max_seq_len - max_new_tokens
    if len(ids) > max_prompt_len:
        print(f"  [警告] プロンプトが長いため末尾 {max_prompt_len} トークンに切り詰めました"
              f" ({len(ids)} → {max_prompt_len})")
        ids = ids[-max_prompt_len:]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            n_loops=n_loops,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    new_ids = output_ids[0, len(ids):].tolist()
    result = tokenizer.decode(new_ids)

    if finance_mode:
        # Stop at next instruction boundary so model doesn't generate new prompts
        stop_idx = result.find(_INSTRUCT_STOP)
        if stop_idx != -1:
            result = result[:stop_idx]

    return result.strip()


# ---------------------------------------------------------------------------
# 対話ループ
# ---------------------------------------------------------------------------

def chat_loop(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Device: {device}\n")

    ckpt_path = args.ckpt or find_latest_ckpt(args.ckpt_dir)
    if not ckpt_path:
        print(f"エラー: {args.ckpt_dir} にチェックポイントが見つかりません。--ckpt で指定してください。")
        sys.exit(1)

    model, cfg = load_model(ckpt_path, device, allow_unsafe=args.allow_unsafe_checkpoint)
    tokenizer = build_tokenizer(cfg.vocab_size, mode=args.tokenizer)

    # top_k の上限を vocab_size に制限
    top_k = min(args.top_k, cfg.vocab_size) if args.top_k > 0 else 0

    print(f"\n設定: temperature={args.temp}  top_k={top_k}  "
          f"max_tokens={args.max_tokens}  loops={args.loops}"
          + ("  [金融指示モード ON]" if args.finance_mode else ""))
    print("─" * 60)
    if args.finance_mode:
        print(_FINANCE_DISCLAIMER)
        print("─" * 60)
    print("プロンプトを入力してください。終了は Ctrl+C または 'quit'")
    print("─" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        if not prompt or prompt.lower() in {"quit", "exit", "q"}:
            print("終了します。")
            break

        print("生成中...", end="\r", flush=True)
        result = generate(
            model, cfg, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=top_k,
            n_loops=args.loops,
            device=device,
            finance_mode=args.finance_mode,
            repetition_penalty=args.rep_penalty,
        )
        print(" " * 20, end="\r")  # "生成中..." の残骸を消去
        print(f"[プロンプト] {prompt}")
        print(f"[生成]      {result}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BushidoMythos 推論チャット")
    p.add_argument("--ckpt",         default=None,
                   help="チェックポイントパス (省略時: --ckpt_dir から自動選択)")
    p.add_argument("--ckpt_dir",     default="checkpoints/finance_a100_v2",
                   help="チェックポイントディレクトリ (phase5_final.pt → phase4_final.pt → phase3_final.pt → final.pt → phase2_final.pt → phase1_final.pt → step_*.pt の順で検索)")
    p.add_argument("--tokenizer",    default="auto", choices=["auto", "gpt2", "mythos"],
                   help="トークナイザ: auto (vocab_size で自動判定) / gpt2 / mythos")
    p.add_argument("--temp",         type=float, default=0.8,
                   help="サンプリング温度 (> 0、低い=確定的、高い=多様)")
    p.add_argument("--top_k",        type=int,   default=50,
                   help="Top-K サンプリング (0=無効)")
    p.add_argument("--max_tokens",   type=int,   default=64,
                   help="最大生成トークン数")
    p.add_argument("--loops",        type=int,   default=4,
                   help="再帰ループ回数 (4=高速・低消費、8=高品質・高精度、max_loop_iters 超えも可)")
    p.add_argument("--finance_mode", action="store_true",
                   help="プロンプトを ### Instruction: / ### Response: 形式に変換し、"
                        "リスク注記 suffix を追加する。"
                        "Phase 3 以降の SFT 済みモデルでは推奨（なしだと生テキスト補完になる）")
    p.add_argument("--rep_penalty",  type=float, default=1.3,
                   help="繰り返しペナルティ (1.0=無効、1.3=推奨、高いほど繰り返しを抑制)")
    p.add_argument("--allow_unsafe_checkpoint", action="store_true",
                   help="weights_only=False で checkpoint をロードする"
                        "（pickle ベース。自分で作成した trusted checkpoint のみ使用）")
    args = p.parse_args()

    # バリデーション
    if args.temp <= 0:
        p.error(f"--temp は 0 より大きい値を指定してください (指定値: {args.temp})")
    if args.top_k < 0:
        p.error(f"--top_k は 0 以上を指定してください (指定値: {args.top_k})")
    if args.max_tokens <= 0:
        p.error(f"--max_tokens は 1 以上を指定してください (指定値: {args.max_tokens})")
    if args.loops <= 0:
        p.error(f"--loops は 1 以上を指定してください (指定値: {args.loops})")
    if args.rep_penalty < 1.0:
        p.error(f"--rep_penalty は 1.0 以上を指定してください (指定値: {args.rep_penalty})")

    return args


if __name__ == "__main__":
    chat_loop(parse_args())
