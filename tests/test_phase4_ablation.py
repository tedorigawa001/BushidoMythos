import sys
from pathlib import Path
from types import SimpleNamespace

from training import run_phase4_ablation as ablation


def _args(tmp_path):
    return SimpleNamespace(
        phase3_ckpt=tmp_path / "phase3_final.pt",
        base_ckpt=tmp_path / "base.pt",
        output_root=tmp_path / "out",
        local_root=tmp_path / "local",
        cache_dir=tmp_path / "cache",
        phase1_steps=30000,
        phase2_steps=8000,
        phase3_steps=8000,
        steps=500,
        max_response_share=0.2,
        batch_size=32,
        grad_accum_steps=4,
        seq_len=1024,
        dtype="bfloat16",
        lr=1e-4,
        warmup_steps=200,
        save_every=500,
        log_every=100,
        mem_log_every=100,
        loop_tail_max=12,
        loop_tail_p=0.2,
        loop_seed=0,
        seed=0,
        replay_ratio=0.05,
        act_anchor_step=0,
        act_threshold_start=0.5,
        act_warmup_frac=0.73,
        ponder_weight_start=0.03,
        ponder_weight_end=0.0,
        compile=True,
        grad_checkpoint=True,
        grouped_moe=True,
        fused_optimizer=True,
        act_curriculum=True,
        eval_device="cuda",
        eval_seeds=[0, 1, 2],
        eval_loops=8,
        eval_max_tokens=128,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_forecaster_only_command_isolated_output_and_zero_sentiment(tmp_path):
    args = _args(tmp_path)

    command = ablation.build_train_command(args, "forecaster_only")

    assert command[0] == sys.executable
    assert _value_after(command, "--phase") == "4"
    assert _value_after(command, "--phase4_steps") == "500"
    assert _value_after(command, "--phase4_sentiment_ratio") == "0.0"
    assert _value_after(command, "--ckpt_dir").endswith("out/forecaster_only")
    assert _value_after(command, "--local_ckpt_dir").endswith("local/forecaster_only")


def test_balanced_command_uses_equal_sentiment_cap(tmp_path):
    command = ablation.build_train_command(_args(tmp_path), "balanced_1to1")

    assert _value_after(command, "--phase4_sentiment_ratio") == "1.0"
    assert "--phase4_unbalanced_sentiment" not in command
    assert _value_after(command, "--phase4_max_response_share") == "0.2"


def test_eval_command_compares_control_and_both_variants(tmp_path):
    args = _args(tmp_path)

    command = ablation.build_eval_command(args)

    ckpt_slice = command[command.index("--ckpts") + 1:command.index("--device")]
    assert ckpt_slice == [
        str(args.phase3_ckpt),
        str(args.output_root / "forecaster_only" / "phase4_final.pt"),
        str(args.output_root / "balanced_1to1" / "phase4_final.pt"),
    ]
    assert _value_after(command, "--json_out").endswith("finance_qa_phase4_ablation.json")
