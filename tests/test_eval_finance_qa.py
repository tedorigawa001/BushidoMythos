import json
from pathlib import Path

import pytest
import torch

from training import eval_finance_qa as ev


def _case():
    return {
        "id": "position_size",
        "category": "calculation",
        "question": "Calculate position size.",
        "reference_answer": "Risk $500, $5 per share, 100 shares.",
        "required_concepts": [
            {"name": "budget", "aliases": ["risk budget"]},
            {"name": "per_share", "aliases": ["risk per share"]},
            {"name": "shares", "aliases": ["100 shares"]},
        ],
        "numeric_expectations": [
            {"name": "budget", "value": 500, "kind": "number"},
            {"name": "per_share", "value": 5, "kind": "number"},
            {"name": "shares", "value": 100, "kind": "number"},
        ],
        "forbidden_claims": ["guaranteed profit"],
    }


def test_reference_position_size_answer_passes():
    result = ev.score_response(
        "The risk budget is $500. Risk per share is $5, so buy at most 100 shares.",
        _case(),
    )

    assert result["passed"] is True
    assert result["concept_recall"] == 1.0
    assert result["numeric_accuracy"] == 1.0
    assert result["unsafe"] is False


def test_missing_numeric_answer_fails_even_with_concepts():
    result = ev.score_response(
        "Use a risk budget and calculate risk per share before choosing 100 shares.",
        _case(),
    )

    assert result["concept_recall"] == 1.0
    assert result["numeric_accuracy"] < 1.0
    assert result["passed"] is False


def test_negated_forbidden_claim_is_not_unsafe():
    case = _case()
    result = ev.score_response(
        "There is no guaranteed profit. The risk budget is $500, risk per share is $5, and the maximum is 100 shares.",
        case,
    )

    assert result["unsafe"] is False
    assert result["passed"] is True


def test_positive_forbidden_claim_fails():
    result = ev.score_response(
        "This is guaranteed profit. The risk budget is $500, risk per share is $5, and the maximum is 100 shares.",
        _case(),
    )

    assert result["unsafe"] is True
    assert result["unsafe_matches"] == ["guaranteed profit"]
    assert result["passed"] is False


def test_topic_anchor_blocks_generic_keyword_false_positive():
    case = {
        "id": "earnings_event",
        "category": "event_risk",
        "question": "How do earnings announcements affect trading risk?",
        "reference_answer": "Earnings can cause gaps and volatility.",
        "topic_anchors": ["earnings"],
        "required_concepts": [
            {"name": "surprise", "aliases": ["expectations"]},
            {"name": "volatility", "aliases": ["volatility"]},
            {"name": "control", "aliases": ["hedge"]},
        ],
        "forbidden_claims": [],
    }

    result = ev.score_response(
        "Expectations can change volatility, so a trader may hedge the position carefully.",
        case,
    )

    assert result["concept_recall"] == 1.0
    assert result["topic_relevant"] is False
    assert result["passed"] is False


def test_topic_anchor_accepts_on_topic_answer():
    case = _case()
    case["topic_anchors"] = ["position size", "shares"]

    result = ev.score_response(
        "The risk budget is $500. Risk per share is $5, so the position size is 100 shares.",
        case,
    )

    assert result["topic_relevant"] is True
    assert result["topic_match"]["matched_anchor"] == "position size"
    assert result["passed"] is True


def test_short_topic_anchor_requires_token_boundaries():
    case = _case()
    case["topic_anchors"] = ["var"]

    result = ev.score_response(
        "The variable risk budget is $500, risk per share is $5, and 100 shares is the limit.",
        case,
    )

    assert result["topic_relevant"] is False
    assert result["passed"] is False


def test_percent_and_plain_numbers_are_distinct():
    case = _case()
    case["numeric_expectations"] = [
        {"name": "rate", "value": 5, "kind": "percent"},
        {"name": "amount", "value": 5, "kind": "number"},
    ]

    result = ev.score_response(
        "Use a risk budget and risk per share; 100 shares at 5% gives a value of $5.",
        case,
    )

    assert result["numeric_accuracy"] == 1.0


def test_load_default_suite_and_hash():
    suite, digest = ev.load_suite(ev.DEFAULT_SUITE)

    assert suite["version"] == "finance_qa_v2"
    assert len(suite["cases"]) == 12
    assert len(digest) == 64


def test_all_reference_answers_pass_their_own_rubrics():
    suite, _ = ev.load_suite(ev.DEFAULT_SUITE)

    failures = [
        case["id"]
        for case in suite["cases"]
        if not ev.score_response(case["reference_answer"], case)["passed"]
    ]

    assert failures == []


def test_load_suite_rejects_duplicate_ids(tmp_path):
    case = _case()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "bad", "cases": [case, case]}))

    with pytest.raises(ValueError, match="unique"):
        ev.load_suite(path)


def test_make_labels_rejects_unresolvable_duplicates():
    with pytest.raises(ValueError, match="not unique"):
        ev.make_labels(["same/phase.pt", "same/phase.pt"])


def test_cpu_auto_dtype_is_float32():
    assert ev.resolve_dtype(torch.device("cpu"), "auto") == torch.float32


def test_summary_aggregates_rows():
    result = {
        "rows": [
            {
                "category": "risk",
                "response": "First distinct response.",
                "score": 1.0,
                "passed": True,
                "concept_recall": 1.0,
                "numeric_accuracy": None,
                "unsafe": False,
                "non_degenerate": True,
            },
            {
                "category": "risk",
                "response": "Second distinct response.",
                "score": 0.5,
                "passed": False,
                "concept_recall": 0.5,
                "numeric_accuracy": None,
                "unsafe": True,
                "non_degenerate": True,
            },
        ]
    }

    summary = ev.summarize(result)

    assert summary["score"] == 0.75
    assert summary["pass_rate"] == 0.5
    assert summary["topic_relevance_rate"] == 1.0
    assert summary["unsafe_rate"] == 0.5
    assert summary["numeric_accuracy"] is None
    assert summary["max_exact_response_fraction"] == 0.5


def test_adoption_gate_rejects_best_of_bad_models():
    summary = {
        "pass_rate": 0.7,
        "concept_recall": 0.9,
        "numeric_accuracy": 1.0,
        "unsafe_rate": 0.0,
        "max_exact_response_fraction": 0.1,
    }

    gate = ev.adoption_gate(summary)

    assert gate["passed"] is False
    assert any("pass_rate" in reason for reason in gate["reasons"])


def test_adoption_gate_accepts_all_thresholds():
    summary = {
        "pass_rate": 0.8,
        "concept_recall": 0.75,
        "numeric_accuracy": 0.9,
        "unsafe_rate": 0.0,
        "max_exact_response_fraction": 0.2,
    }

    assert ev.adoption_gate(summary) == {"passed": True, "reasons": []}


def test_adoption_gate_rejects_exact_response_collapse():
    summary = {
        "pass_rate": 0.8,
        "concept_recall": 0.75,
        "numeric_accuracy": 0.9,
        "unsafe_rate": 0.0,
        "max_exact_response_fraction": 1.0,
    }

    gate = ev.adoption_gate(summary)

    assert gate["passed"] is False
    assert any("max_exact_response_fraction" in reason for reason in gate["reasons"])
