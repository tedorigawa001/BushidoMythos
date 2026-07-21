import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import run_phase5_validation as runner


def _args(tmp_path):
    return SimpleNamespace(
        phase3_ckpt=tmp_path / "phase3.pt",
        validation_data=tmp_path / "validation.json",
        held_out_suite=tmp_path / "held_out.json",
        output_root=tmp_path / "out",
        eval_device="cuda",
        dtype="bfloat16",
        eval_batch_size=16,
        seq_len=256,
        eval_loops=8,
        min_nll_improvement=0.05,
        min_binding_accuracy=0.60,
        max_category_regression=0.10,
        final_seeds=[0, 1, 2],
        final_max_tokens=128,
        allow_unsafe_checkpoint=False,
    )


def _value_after(command, flag):
    return command[command.index(flag) + 1]


def test_validation_command_never_receives_final_held_out_suite(tmp_path):
    args = _args(tmp_path)
    candidates = [tmp_path / "step_046050.pt", tmp_path / "step_046100.pt"]

    command = runner.build_validation_command(args, candidates)

    assert command[0] == sys.executable
    assert command[1] == "training/eval_finance_validation.py"
    assert "--validation_data" in command
    assert str(args.validation_data) in command
    assert str(args.held_out_suite) not in command
    ckpts = command[command.index("--ckpts") + 1:command.index("--validation_data")]
    assert ckpts == [str(args.phase3_ckpt), *map(str, candidates)]


def test_final_command_uses_only_baseline_and_selected_checkpoint(tmp_path):
    args = _args(tmp_path)
    selected = tmp_path / "step_046100.pt"

    command = runner.build_final_command(args, selected)

    assert command[1] == "training/eval_finance_qa.py"
    ckpts = command[command.index("--ckpts") + 1:command.index("--suite")]
    assert ckpts == [str(args.phase3_ckpt), str(selected)]
    assert _value_after(command, "--suite") == str(args.held_out_suite)


def test_candidate_checkpoints_are_periodic_then_final(tmp_path):
    (tmp_path / "step_046100.pt").write_bytes(b"")
    (tmp_path / "step_046050.pt").write_bytes(b"")
    (tmp_path / "phase5_final.pt").write_bytes(b"")

    result = runner.candidate_checkpoints(tmp_path)

    assert [path.name for path in result] == [
        "step_046050.pt", "step_046100.pt", "phase5_final.pt",
    ]


def test_parser_rejects_notebook_wide_sequence_length(monkeypatch, tmp_path):
    paths = {
        name: tmp_path / name
        for name in ("phase3.pt", "base.pt", "train.json", "validation.json", "held.json")
    }
    for path in paths.values():
        path.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", [
        "run_phase5_validation.py",
        "--phase3_ckpt", str(paths["phase3.pt"]),
        "--base_ckpt", str(paths["base.pt"]),
        "--curated_data", str(paths["train.json"]),
        "--validation_data", str(paths["validation.json"]),
        "--held_out_suite", str(paths["held.json"]),
        "--output_root", str(tmp_path / "out"),
        "--seq_len", "1024",
    ])

    with pytest.raises(SystemExit, match="2"):
        runner.parse_args()


def test_run_checked_repeats_child_error_tail():
    command = [
        sys.executable, "-c",
        "print('diagnostic root cause', flush=True); raise RuntimeError('child failed')",
    ]

    with pytest.raises(RuntimeError, match="diagnostic root cause"):
        runner.run_checked(command, "test child")
