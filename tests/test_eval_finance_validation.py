from pathlib import Path
from types import SimpleNamespace

import torch

from training import eval_finance_validation as validation


ROOT = Path(__file__).parent.parent


def _summary(nll, prompt_binding, calculation_binding, category_nll=None):
    return {
        "response_nll": nll,
        "prompt_binding": {"accuracy": prompt_binding},
        "calculation_binding": {"accuracy": calculation_binding},
        "category_nll": category_nll or {"calculation": nll, "risk": nll},
    }


def test_binding_rows_use_different_response_from_same_family():
    payload, sha256 = validation.load_validation(validation.DEFAULT_VALIDATION)

    rows = validation.build_binding_rows(payload["examples"])

    assert len(rows) == 160
    assert len(sha256) == 64
    assert all(row["id"] != row["distractor_id"] for row in rows)
    family_by_id = {item["id"]: item["scenario_family"] for item in payload["examples"]}
    assert all(
        family_by_id[row["id"]] == family_by_id[row["distractor_id"]]
        for row in rows
    )
    assert sum(row["is_calculation"] for row in rows) == 20


def test_resolve_device_calls_chat_auto_detector_without_arguments(monkeypatch):
    calls = []

    def fake_get_device():
        calls.append(True)
        return torch.device("cpu")

    monkeypatch.setattr(validation, "get_device", fake_get_device)

    assert validation.resolve_device("auto") == torch.device("cpu")
    assert calls == [True]
    assert validation.resolve_device("cuda") == torch.device("cuda")
    assert calls == [True]


def test_checkpoint_gate_requires_nll_binding_and_category_guardrails():
    baseline = _summary(2.0, 0.55, 0.50)
    candidate = _summary(1.8, 0.70, 0.65, {"calculation": 1.9, "risk": 1.8})

    gate = validation.checkpoint_gate(baseline, candidate, 0.05, 0.60, 0.10)

    assert gate == {"passed": True, "reasons": []}


def test_checkpoint_gate_rejects_low_binding_even_when_nll_improves():
    baseline = _summary(2.0, 0.55, 0.50)
    candidate = _summary(1.5, 0.70, 0.40, {"calculation": 1.5, "risk": 1.5})

    gate = validation.checkpoint_gate(baseline, candidate, 0.05, 0.60, 0.10)

    assert gate["passed"] is False
    assert any("calculation_binding_accuracy" in reason for reason in gate["reasons"])


def test_select_checkpoint_chooses_lowest_nll_among_eligible():
    results = [
        {"label": "phase3", "checkpoint": "phase3.pt", "summary": _summary(2.0, 0.5, 0.5)},
        {"label": "step50", "checkpoint": "step50.pt", "summary": _summary(1.8, 0.7, 0.7)},
        {"label": "step100", "checkpoint": "step100.pt", "summary": _summary(1.6, 0.8, 0.8)},
    ]
    args = SimpleNamespace(
        min_nll_improvement=0.05,
        min_binding_accuracy=0.60,
        max_category_regression=0.10,
    )

    selected, gates = validation.select_checkpoint(results, args)

    assert selected == "step100.pt"
    assert all(gate["passed"] for gate in gates.values())


def test_tokenized_evaluation_counts_only_response_tokens():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [min(ord(char), 31) for char in text]

    class UniformModel:
        def __call__(self, x, n_loops=None):
            return torch.zeros((*x.shape, 32), dtype=torch.float32)

    rows = [{
        "id": "example", "category": "calculation", "scenario_family": "family",
        "is_calculation": True, "instruction": "Q", "context": "",
        "correct_response": "right", "distractor_id": "other",
        "distractor_response": "wrong",
    }]
    tokenized = validation._tokenize_rows(rows, Tokenizer(), 32, seq_len=32)
    args = SimpleNamespace(batch_size=2, loops=1)

    evaluated = validation._evaluate_tokenized(
        UniformModel(), tokenized, args, torch.device("cpu"), torch.float32
    )

    assert len(evaluated) == 2
    assert all(row["response_tokens"] > 0 for row in evaluated)
    assert all(abs(row["nll"] - torch.log(torch.tensor(32.0)).item()) < 1e-6 for row in evaluated)
