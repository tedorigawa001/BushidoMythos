import argparse
import hashlib
import json
from pathlib import Path

import pytest

from training import build_finance_qa_curated_v2 as corpus


ROOT = Path(__file__).parent.parent


def test_default_corpus_is_balanced_family_disjoint_and_calculation_checked():
    train = corpus.build_examples("train", 40)
    validation = corpus.build_examples("validation", 10)

    manifest = corpus.validate_corpus(train, validation, corpus.DEFAULT_HELD_OUT)

    assert manifest["counts"] == {"train": 640, "validation": 160, "held_out": 12}
    assert set(manifest["categories"]["train"].values()) == {80}
    assert set(manifest["categories"]["validation"].values()) == {20}
    assert manifest["calculation_examples"] == 100
    families = manifest["scenario_families"]
    assert not (set(families["train"]) & set(families["validation"]))
    assert not (set(families["train"]) & set(families["held_out"]))
    assert not (set(families["validation"]) & set(families["held_out"]))
    assert all(manifest["checks"].values())


def test_committed_corpus_matches_generator():
    expected = {
        corpus.DEFAULT_TRAIN: corpus._payload("train", corpus.build_examples("train", 40)),
        corpus.DEFAULT_VALIDATION: corpus._payload(
            "validation", corpus.build_examples("validation", 10)
        ),
    }

    for path, payload in expected.items():
        assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_generated_split_bytes_are_reproducible(tmp_path):
    hashes = []
    for run in ("first", "second"):
        root = tmp_path / run
        args = argparse.Namespace(
            train_out=root / "train.json",
            validation_out=root / "validation.json",
            held_out=corpus.DEFAULT_HELD_OUT,
            manifest_out=root / "manifest.json",
            train_per_family=4,
            validation_per_family=2,
        )
        corpus.build_and_write(args)
        hashes.append((
            hashlib.sha256(args.train_out.read_bytes()).hexdigest(),
            hashlib.sha256(args.validation_out.read_bytes()).hexdigest(),
        ))

    assert hashes[0] == hashes[1]


def test_validation_rejects_family_leakage():
    train = corpus.build_examples("train", 1)
    validation = corpus.build_examples("validation", 1)
    validation[0]["scenario_family"] = train[0]["scenario_family"]

    with pytest.raises(ValueError, match="scenario-family leakage"):
        corpus.validate_corpus(train, validation, corpus.DEFAULT_HELD_OUT)
