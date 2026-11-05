"""Behavioral checks for CTC sequence metrics and held-out confidence scaling."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from anpr.alphabet import CHARACTERS
from anpr.ctc_evaluate import (
    character_edit_distance,
    evaluate_ctc_logits,
    fit_ctc_temperature,
    fit_validation_ctc_temperature,
    load_ctc_calibration,
    main,
)


def frame_logits(path, strength=12.0):
    """Represent a controlled CTC path; underscore denotes the blank class."""
    # Strong but finite alternatives keep exact target likelihoods representable.
    logits = [[-strength] * 37 for _ in path]
    for frame, character in enumerate(path):
        # Repeated visible characters collapse unless an intervening blank separates them.
        logits[frame][0 if character == "_" else CHARACTERS.index(character) + 1] = strength
    return logits


def test_repeated_letters_use_ctc_sequence_metrics_without_positional_claims():
    """A repeated registration must survive collapse and retain high sequence likelihood."""
    # Both paths represent two separate As, despite several adjacent nonblank frames.
    metrics = evaluate_ctc_logits(
        [frame_logits("A_A"), frame_logits("AA_A")], ["AA", "AA"], budgets=(1, 5), max_length=3, beam_width=10
    )
    assert metrics["sample_count"] == 2
    assert metrics["accuracy_exact_plate"] == 1
    assert metrics["true_plate_recall_at_k"] == {"1": 1, "5": 1}
    assert metrics["character_error_rate"] == 0
    assert metrics["nll_true_plate_ctc"] < 1e-6
    assert metrics["ece_top1_sequence_score"] < 1e-6
    assert metrics["whole_plate_confidence_calibrated"] is False
    assert "accuracy_position_including_padding" not in metrics
    assert "ece_character_nonpadding" not in metrics


def test_character_error_rate_counts_insertions_and_can_exceed_one():
    """CER uses ground-truth character count rather than clipping a bad prediction."""
    # Four predicted As versus one true A gives three insertions per true character.
    metrics = evaluate_ctc_logits([frame_logits("A_A_A_A")], ["A"], budgets=(1,), max_length=7, beam_width=10)
    assert metrics["character_error_rate"] == 3
    assert character_edit_distance("AAAA", "A") == 3
    assert character_edit_distance("", "AB") == 2
    assert character_edit_distance("AB", "BA") == 2


def test_empty_or_unreadable_predictions_remain_in_all_denominators():
    """An abstention must count as missed plate characters and empty candidate lists."""
    # A decoder abstention with no suggestions is valid output, not a row to drop.
    decoded = {"candidates": [], "status": "unreadable"}
    with mock.patch("anpr.ctc_evaluate.decode_ctc_logits", return_value=decoded):
        metrics = evaluate_ctc_logits(
            [frame_logits("A_A")], ["AA"], budgets=(1, 5), max_length=3, beam_width=10, latency_ms=[25.0]
        )
    assert metrics["sample_count"] == 1
    assert metrics["accuracy_exact_plate"] == 0
    assert metrics["true_plate_recall_at_k"] == {"1": 0, "5": 0}
    assert metrics["mean_candidate_count_at_k"] == {"1": 0, "5": 0}
    assert metrics["character_error_rate"] == 1
    assert metrics["status_counts"] == {"unreadable": 1}
    assert metrics["latency_preprocess_model_mean_ms"] == 25


def test_metrics_reject_impossible_repeated_character_alignment():
    """Impossible targets must fail explicitly instead of exporting infinite JSON NLL."""
    with pytest.raises(ValueError, match="align|frames"):
        evaluate_ctc_logits([frame_logits("AA")], ["AA"], budgets=(1,), max_length=3, beam_width=10)


@pytest.mark.parametrize(
    "change",
    [
        {"budgets": ()},
        {"budgets": (0,)},
        {"beam_width": 1, "budgets": (5,)},
        {"temperature": 0},
        {"latency_ms": []},
        {"max_length": 0},
    ],
)
def test_invalid_metric_configuration_is_rejected(change):
    """Invalid denominators and search budgets must not create partial metric reports."""
    # Each override violates a public configuration contract independently of recognition accuracy.
    options = {"budgets": (1,), "max_length": 3, "beam_width": 10, **change}
    with pytest.raises(ValueError):
        evaluate_ctc_logits([frame_logits("A_A")], ["AA"], **options)


def test_validation_temperature_grid_reduces_overconfident_sequence_nll():
    """A wrong sharp path should prefer a softer scale on independent validation labels."""
    # The visual logits strongly prefer A while the validation annotation says B.
    logits = [frame_logits("A", strength=5)]
    fitted_temperature = fit_ctc_temperature(logits, ["B"], temperatures=(0.5, 1, 2, 4), max_length=3)
    assert fitted_temperature == 4
    assert fitted_temperature in (0.5, 1, 2, 4)


def test_calibration_predictor_receives_validation_records_only():
    """Target-test labels cannot enter the temperature fitting callback."""
    # Absolute paths and independent identity/source groups reproduce the manifest boundary.
    splits = {
        "train": [SimpleNamespace(path=Path("/train.png"), plate="A", group="train")],
        "val": [SimpleNamespace(path=Path("/val.png"), plate="B", group="val")],
        "test": [SimpleNamespace(path=Path("/test.png"), plate="C", group="test")],
    }
    predictor = mock.Mock(return_value=frame_logits("A"))
    fitted_temperature = fit_validation_ctc_temperature(splits, predictor, max_length=3, temperatures=(1, 2))
    assert fitted_temperature == 2
    predictor.assert_called_once_with(splits["val"][0])
    # Renamed copies of the same plate must still prevent validation/test contamination.
    splits["test"][0].plate = "B"
    with pytest.raises(ValueError, match="leakage|overlap"):
        fit_validation_ctc_temperature(splits, predictor, max_length=3)


def test_frozen_calibration_reuses_shared_membership_checks_and_requires_ctc_kind(tmp_path):
    """A positional-head temperature cannot silently become a CTC confidence scale."""
    # These bytes identify a checkpoint without loading a neural network in a pure helper test.
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"ctc weights")
    calibration_path = tmp_path / "scale.json"
    calibration = {
        "temperature": 2,
        "model_kind": "ctc",
        "fit_split": "val",
        "validation_sample_count": 1,
        "validation_provenance": [{"plate": "B", "group": "val", "path": "/val.png"}],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": "a" * 64,
        "whole_plate_confidence_calibrated": False,
    }
    calibration_path.write_text(json.dumps(calibration))
    with mock.patch("anpr.ctc_evaluate.fit_ctc_temperature", side_effect=AssertionError("refit")):
        assert load_ctc_calibration(calibration_path, checkpoint_path) == calibration
    calibration_path.write_text(json.dumps({**calibration, "model_kind": "positional"}))
    with pytest.raises(ValueError, match="kind|CTC|ctc"):
        load_ctc_calibration(calibration_path, checkpoint_path)


def test_cli_path_ownership_is_checked_before_loading_any_model(tmp_path):
    """A report typo must not overwrite the only checkpoint or begin model deserialization."""
    # Invalid torch bytes make the early ownership check distinguishable from model IO.
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint_path.write_bytes(b"trusted weights")
    with pytest.raises(ValueError, match="alias|input"):
        main(
            [
                "--checkpoint",
                str(checkpoint_path),
                "--manifest",
                str(tmp_path / "manifest.csv"),
                "--output",
                str(checkpoint_path),
            ]
        )
    assert checkpoint_path.read_bytes() == b"trusted weights"
