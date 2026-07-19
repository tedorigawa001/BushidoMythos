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
            assert "proc.check_returncode()" in src, (
                f"Training cell {i} does not propagate subprocess failure"
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

    def test_dependency_install_fails_loudly_and_checks_bitsandbytes(self):
        install = next(
            cell for cell in self.nb["cells"] if cell.get("id") == "install-deps"
        )
        source = "".join(install["source"])
        assert "subprocess.run" in source
        assert "check=True" in source
        assert "import bitsandbytes" in source

    def test_save_cache_cell_exists(self):
        """Notebook must contain a save-back cell after training."""
        all_src = " ".join(
            "".join(c["source"]) for c in self.nb["cells"]
        )
        assert "shutil.copytree" in all_src, (
            "Cache save-back (shutil.copytree) not found in notebook"
        )

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
        inference = ids.index("section-chat")
        assert grouped_training < chat_benchmark < inference


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
