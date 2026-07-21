"""
Consistency tests for colab_finance_train.ipynb.

Verifies that all training cells contain the required flags and that the
notebook structure (section numbers, cell order) is coherent.  This acts
as a regression guard — any accidental omission of a flag from a cell
will be caught before the notebook is run on Colab.
"""

import json
import re
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

NOTEBOOK = repo_root / "colab_finance_train.ipynb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_nb():
    with open(NOTEBOOK) as f:
        return json.load(f)


def _training_cells(nb):
    """Return code cells that build a subprocess cmd list (training cells)."""
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if '"--log_every"' in src and "subprocess" in src:
            cells.append(src)
    return cells


# ---------------------------------------------------------------------------
# Required-flag tests
# ---------------------------------------------------------------------------

class TestRequiredFlags:
    """Every training cell must contain the expected flags."""

    def setup_method(self):
        nb = _load_nb()
        self.cells = _training_cells(nb)

    def test_at_least_one_training_cell_found(self):
        assert len(self.cells) >= 1, "No training cells found in notebook"

    @pytest.mark.parametrize("flag", [
        "--cache_dir",
        "--mem_log_every",
        "--log_every",
        "--grad_accum_steps",
        "--batch_size",
        "--seq_len",
        "--dtype",
        "--save_every",
        "--log_file",
        "--local_ckpt_dir",
    ])
    def test_flag_present_in_all_training_cells(self, flag):
        missing = [i for i, src in enumerate(self.cells) if flag not in src]
        assert not missing, (
            f"Flag '{flag}' missing from training cell(s) at index {missing}"
        )

    def test_cache_dir_points_to_content_cache(self):
        """--cache_dir value must be /content/cache in every training cell."""
        for i, src in enumerate(self.cells):
            assert "/content/cache" in src, (
                f"Training cell {i} has --cache_dir but not '/content/cache'"
            )

    def test_grad_checkpoint_conditional_in_all_training_cells(self):
        """Every training cell must have the USE_GRAD_CHECKPOINT conditional."""
        for i, src in enumerate(self.cells):
            assert "USE_GRAD_CHECKPOINT" in src, (
                f"Training cell {i} missing USE_GRAD_CHECKPOINT conditional"
            )

    def test_grouped_moe_conditional_in_all_training_cells(self):
        """Every training phase must apply the production grouped MoE switch."""
        for i, src in enumerate(self.cells):
            assert 'if GROUPED_MOE:' in src and 'cmd.append("--grouped_moe")' in src, (
                f"Training cell {i} missing GROUPED_MOE conditional"
            )

    def test_liger_fused_ce_conditional_in_all_training_cells(self):
        """Every training phase must honor the benchmark-gated Liger switch."""
        for i, src in enumerate(self.cells):
            assert (
                'if LIGER_FUSED_CE:' in src
                and 'cmd.append("--liger_fused_ce")' in src
            ), f"Training cell {i} missing LIGER_FUSED_CE conditional"

    def test_chunked_ce_conditional_in_all_training_cells(self):
        """Every training phase must honor the OOM fallback chunk size."""
        all_source = "".join(
            "".join(cell["source"]) for cell in _load_nb()["cells"]
        )
        assert "LIGER_FUSED_CE and CE_CHUNK_SIZE > 0" in all_source
        for i, src in enumerate(self.cells):
            assert (
                "if CE_CHUNK_SIZE > 0:" in src
                and 'cmd += ["--ce_chunk_size", str(CE_CHUNK_SIZE)]' in src
            ), f"Training cell {i} missing CE_CHUNK_SIZE conditional"

    def test_optimizer_switches_are_exclusive_and_present_in_all_cells(self):
        all_source = "".join(
            "".join(cell["source"]) for cell in _load_nb()["cells"]
        )
        assert "OPTIM8BIT = not FUSED_OPTIMIZER" in all_source
        for i, src in enumerate(self.cells):
            assert 'if OPTIM8BIT:' in src and 'cmd.append("--optim8bit")' in src, (
                f"Training cell {i} missing OPTIM8BIT conditional"
            )
            assert (
                'if FUSED_OPTIMIZER:' in src
                and 'cmd.append("--fused_optimizer")' in src
            ), f"Training cell {i} missing FUSED_OPTIMIZER conditional"

    def test_subprocess_failure_propagates_in_all_training_cells(self):
        """A failed training process must fail its Colab cell."""
        for i, src in enumerate(self.cells):
            assert "raise subprocess.CalledProcessError(proc.returncode, cmd)" in src, (
                f"Training cell {i} does not propagate subprocess failure"
            )

    def test_act_curriculum_flag_names_are_exact_in_all_training_cells(self):
        expected = (
            "--act_anchor_step",
            "--act_threshold_start",
            "--act_warmup_frac",
            "--ponder_weight_start",
            "--ponder_weight_end",
        )
        for i, src in enumerate(self.cells):
            missing = [flag for flag in expected if f'"{flag}"' not in src]
            assert not missing, (
                f"Training cell {i} has missing or corrupted ACT flags: {missing}"
            )


# ---------------------------------------------------------------------------
# Notebook structure tests
# ---------------------------------------------------------------------------

class TestNotebookStructure:

    def setup_method(self):
        self.nb = _load_nb()

    def test_notebook_file_exists(self):
        assert NOTEBOOK.exists()

    def test_no_character_split_source(self):
        """Every cell source must be a list of strings, not individual chars."""
        for i, cell in enumerate(self.nb["cells"]):
            src = cell["source"]
            if src:
                assert len(src[0]) > 1 or len(src) == 1, (
                    f"Cell {i} source appears character-split: {src[:5]}"
                )

    def test_markdown_section_numbers_unique(self):
        """Top-level '## N.' headings must not duplicate (e.g. two '## 2.')."""
        pattern = re.compile(r"^## (\d+)\.")
        seen = {}
        for i, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] != "markdown":
                continue
            first = "".join(cell["source"]).split("\n")[0]
            m = pattern.match(first)
            if m:
                num = m.group(1)
                assert num not in seen, (
                    f"Section number '{num}' appears in both cell {seen[num]}"
                    f" and cell {i}"
                )
                seen[num] = i

    def test_cache_setup_cell_exists(self):
        """Notebook must contain a cache-setup cell (1a or similar)."""
        all_src = " ".join(
            "".join(c["source"]) for c in self.nb["cells"]
        )
        assert "LOCAL_CACHE" in all_src, "Cache setup cell not found"
        assert "DRIVE_CACHE" in all_src, "Drive cache variable not found"

    def test_full_training_uses_a_fresh_dynamic_run_directory(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        gpu_settings = sources["check-gpu"]
        assert 'FRESH_RUN_NAME = "finance_a100_v3_full"' in gpu_settings
        assert "CKPT_SUBDIR = FRESH_RUN_NAME" in gpu_settings
        assert all(
            "finance_a100_v2" not in source for source in sources.values()
        ), "Notebook source must not resume or evaluate the completed v2 run"
        for src in _training_cells(self.nb):
            assert 'f"{REPO}/checkpoints/{CKPT_SUBDIR}"' in src

    def test_finance_qa_compares_phase2_through_phase5(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-finance-qa-eval"]
        assert "training/eval_finance_qa.py" in source
        assert "for phase in (2, 3)" in source
        assert "for phase in (4, 5)" in source
        assert "if path.is_file()" in source
        assert '"--seeds", "0", "1", "2"' in source
        assert "subprocess.run(cmd, check=True)" in source
        assert "finance_qa_phase2_5.json" in source
        assert "finance_qa_phase2_5.md" in source

    def test_phase4_training_has_response_collapse_controls(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        settings = sources["check-gpu"]
        phase4 = sources["run-phase4"]
        assert "PHASE4_STEPS = 0" in settings
        assert "PHASE4_SENTIMENT_RATIO = 0.0" in settings
        assert "PHASE4_MAX_RESPONSE_SHARE = 0.20" in settings
        assert '"--phase4_sentiment_ratio"' in phase4
        assert '"--phase4_max_response_share"' in phase4
        assert "PHASE5_STEPS = 0" in settings
        assert 'PHASE5_DATA_MODE = "curated"' in settings

    def test_phase5_skips_disabled_phase4(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-phase5"]
        assert "if PHASE4_STEPS > 0" in source
        assert 'else f"{CKPT_DIR}/phase3_final.pt"' in source
        assert '"--phase4_steps", str(PHASE4_STEPS)' in source

    def test_phase4_ablation_uses_isolated_runner(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-phase4-ablation"]
        assert "training/run_phase4_ablation.py" in source
        assert '"--steps", "500"' in source
        assert "phase3_final.pt" in source
        assert "subprocess.run(cmd, check=True)" in source
        assert "finance_qa_phase4_ablation.json" in source

    def test_phase5_pilot_uses_audited_isolated_runner(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-phase5-pilot"]
        assert "training/run_phase5_pilot.py" in source
        assert '"--steps", "500"' in source
        assert "phase3_final.pt" in source
        assert "phase5_data_audit.json" in source
        assert "subprocess.run(cmd, check=True)" in source
        assert "finance_qa_phase5_pilot.json" in source

    def test_phase5_step_ablation_rescores_existing_doses_with_v2(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-phase5-step-ablation"]
        assert "training/run_phase5_step_ablation.py" in source
        assert '"--step_variants", "10", "50", "200"' in source
        assert "phase3_final.pt" in source
        assert "historical_phase5" in source
        assert '"--eval_only"' in source
        assert "training/eval_data/finance_qa_v2.json" in source
        assert "subprocess.run(cmd, check=True)" in source
        assert "finance_qa_phase5_step_ablation_v2.json" in source

    def test_large_curated_v2_cell_builds_and_checks_manifest(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["build-finance-curated-v2"]
        assert "training/build_finance_qa_curated_v2.py" in source
        assert '"train": 640' in source
        assert '"validation": 160' in source
        assert '"held_out": 12' in source
        assert 'all(manifest["checks"].values())' in source

    def test_phase5_validation_cell_keeps_final_held_out_behind_gate(self):
        sources = {
            cell.get("id"): "".join(cell["source"])
            for cell in self.nb["cells"]
        }
        source = sources["run-phase5-validation-selection"]
        assert "training/run_phase5_validation.py" in source
        assert '"--save_every", "50"' in source
        assert '"--seq_len", "256"' in source
        assert '"--seq_len", str(SEQ_LEN)' not in source
        assert "completed_phase5.is_file()" in source
        assert 'cmd.append("--skip_train")' in source
        assert "finance_validation_selection.json" in source
        assert "finance_qa_final_selected.json" in source

    def test_dependency_install_fails_loudly_and_checks_bitsandbytes(self):
        install = next(
            cell for cell in self.nb["cells"] if cell.get("id") == "install-deps"
        )
        source = "".join(install["source"])
        assert "subprocess.run" in source
        assert "check=True" in source
        assert "import bitsandbytes" in source
        assert "liger-kernel" in source
        assert "LigerFusedLinearCrossEntropyLoss" in source

    def test_save_cache_cell_exists(self):
        """Notebook must contain a save-back cell after training."""
        all_src = " ".join(
            "".join(c["source"]) for c in self.nb["cells"]
        )
        assert "shutil.copytree" in all_src, (
            "Cache save-back (shutil.copytree) not found in notebook"
        )

    def test_wall_clock_report_cell_precedes_cache_copy(self):
        ids = [cell.get("id") for cell in self.nb["cells"]]
        report = ids.index("run-wall-clock-report")
        cache_copy = ids.index("HsHE3VbzJ3Sb")
        source = "".join(self.nb["cells"][report]["source"])
        assert "training/report_wall_clock.py" in source
        assert "subprocess.run(cmd, check=True)" in source
        assert report < cache_copy

    def test_inference_cell_exists(self):
        """Notebook must have an inference test cell."""
        sources = ["".join(c["source"]) for c in self.nb["cells"]]
        has_inference = any("model.generate" in s or "chat(" in s
                            for s in sources)
        assert has_inference, "No inference/chat cell found"

    def test_chat_grouped_benchmark_precedes_inference_section(self):
        ids = [cell.get("id") for cell in self.nb["cells"]]
        grouped_training = ids.index("run-grouped-moe-benchmark")
        chat_benchmark = ids.index("run-chat-grouped-moe-benchmark")
        native_gqa = ids.index("run-native-gqa-benchmark")
        optimizer = ids.index("run-optimizer-benchmark")
        act_skip = ids.index("run-act-skip-benchmark")
        liger_ce = ids.index("run-liger-fused-ce-benchmark")
        inference = ids.index("section-chat")
        assert (
            grouped_training
            < chat_benchmark
            < native_gqa
            < optimizer
            < act_skip
            < liger_ce
            < inference
        )


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
