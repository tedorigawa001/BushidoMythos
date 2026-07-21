import sys
from pathlib import Path
from types import SimpleNamespace

from training import run_phase5_pilot as pilot


def _args(tmp_path):
    return SimpleNamespace(
        phase3_ckpt=tmp_path / "phase3_final.pt",
        base_ckpt=tmp_path / "base.pt",
        comparison_ckpts=[tmp_path / "old_phase5.pt"],
        output_root=tmp_path / "out",
        local_root=tmp_path / "local",
        cache_dir=tmp_path / "cache",
        curated_data=tmp_path / "curated.json",
        validation_data=tmp_path / "validation.json",
        eval_suite=tmp_path / "eval.json",
        phase1_steps=30000,
        phase2_steps=8000,
        phase3_steps=8000,
        steps=500,
        max_similarity=0.8,
        max_response_share=0.1,
        batch_size=32,
        grad_accum_steps=4,
        seq_len=256,
        dtype="bfloat16",
        lr=2e-5,
        warmup_steps=50,
        save_every=500,
        log_every=50,
        mem_log_every=50,
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


def test_train_command_starts_phase5_directly_from_phase3(tmp_path):
    args = _args(tmp_path)

    command = pilot.build_train_command(args)

    assert command[0] == sys.executable
    assert _value_after(command, "--phase") == "5"
    assert _value_after(command, "--resume") == str(args.phase3_ckpt)
    assert _value_after(command, "--phase4_steps") == "0"
    assert _value_after(command, "--phase5_steps") == "500"
    assert _value_after(command, "--phase5_data_mode") == "curated"
    assert _value_after(command, "--phase5_curated_path") == str(args.curated_data)
    assert _value_after(command, "--phase5_validation_path") == str(
        args.validation_data
    )
    assert _value_after(command, "--phase5_eval_suite") == str(args.eval_suite)
    assert _value_after(command, "--phase5_audit_json").endswith(
        "out/curated/phase5_data_audit.json"
    )
    assert _value_after(command, "--keep_last_n_steps") == "1"


def test_eval_command_compares_control_historical_and_curated(tmp_path):
    args = _args(tmp_path)

    command = pilot.build_eval_command(args)

    ckpts = command[command.index("--ckpts") + 1:command.index("--suite")]
    assert ckpts == [
        str(args.phase3_ckpt),
        str(args.comparison_ckpts[0]),
        str(args.output_root / "curated" / "phase5_final.pt"),
    ]
    assert _value_after(command, "--json_out").endswith("finance_qa_phase5_pilot.json")
