import sys
from pathlib import Path
from types import SimpleNamespace

from training import run_phase5_step_ablation as ablation


def _args(tmp_path):
    return SimpleNamespace(
        phase3_ckpt=tmp_path / "phase3.pt",
        base_ckpt=tmp_path / "base.pt",
        comparison_ckpts=[tmp_path / "aligned500.pt"],
        output_root=tmp_path / "out",
        local_root=tmp_path / "local",
        cache_dir=tmp_path / "cache",
        curated_data=tmp_path / "curated.json",
        validation_data=tmp_path / "validation.json",
        eval_suite=tmp_path / "eval.json",
        step_variants=[10, 50, 200],
        warmup_steps=50,
        phase1_steps=30000,
        phase2_steps=8000,
        phase3_steps=8000,
        batch_size=16,
        grad_accum_steps=8,
        seq_len=256,
        dtype="bfloat16",
        lr=2e-5,
        save_every=500,
        log_every=10,
        mem_log_every=10,
        max_similarity=0.8,
        max_response_share=0.1,
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
        eval_device="cuda",
        eval_seeds=[0, 1, 2],
        eval_loops=8,
        eval_max_tokens=128,
        eval_report_stem="finance_qa_phase5_step_ablation",
        eval_only=False,
        compile=True,
        grad_checkpoint=True,
        grouped_moe=True,
        fused_optimizer=True,
        act_curriculum=True,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_train_commands_isolate_each_step_variant(tmp_path):
    commands = ablation.build_train_commands(_args(tmp_path))

    assert len(commands) == 3
    for steps, command in zip((10, 50, 200), commands):
        name = ablation.variant_name(steps)
        assert command[0] == sys.executable
        assert _value_after(command, "--phase5_steps") == str(steps)
        assert _value_after(command, "--warmup_steps") == "50"
        assert _value_after(command, "--ckpt_dir").endswith(f"out/{name}")
        assert _value_after(command, "--local_ckpt_dir").endswith(f"local/{name}")


def test_eval_command_compares_control_existing_and_all_doses(tmp_path):
    args = _args(tmp_path)
    command = ablation.build_eval_command(args)

    ckpts = command[command.index("--ckpts") + 1:command.index("--suite")]
    assert ckpts == [
        str(args.phase3_ckpt),
        str(args.comparison_ckpts[0]),
        *[
            str(args.output_root / ablation.variant_name(steps) / "phase5_final.pt")
            for steps in args.step_variants
        ],
    ]
    assert _value_after(command, "--json_out").endswith(
        "finance_qa_phase5_step_ablation.json"
    )


def test_eval_command_supports_versioned_report_name(tmp_path):
    args = _args(tmp_path)
    args.eval_report_stem = "finance_qa_phase5_step_ablation_v2"

    command = ablation.build_eval_command(args)

    assert _value_after(command, "--json_out").endswith(
        "finance_qa_phase5_step_ablation_v2.json"
    )
    assert _value_after(command, "--md_out").endswith(
        "finance_qa_phase5_step_ablation_v2.md"
    )
