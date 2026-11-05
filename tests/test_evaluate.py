"""Behavioral checks for ranking metrics and validation-only calibration."""

import math
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from anpr.alphabet import CHARACTERS
from anpr.evaluate import (
    evaluate_logits,
    fit_temperature,
    fit_validation_temperature,
    load_calibration,
    main,
    positional_label_coverage,
)


def test_positional_label_coverage_detects_unseen_characters_and_padding():
    """Vocabulary support alone does not imply positive training exposure at every head."""
    # Head one sees A/B/1, while the final head sees padding only in these training labels.
    training_rows = [{"plate": plate} for plate in ["AA", "AB", "A1"]]
    coverage = positional_label_coverage(training_rows, ["AA", "AC", "AA1"], max_length=3)
    assert coverage["training_class_count_by_position"] == [1, 3, 1]
    assert coverage["test_plates_with_unseen_position_class"] == 2
    assert coverage["test_unseen_class_count_by_position"] == [0, 1, 1]
    assert coverage["test_sample_count"] == 3
    assert coverage["includes_trailing_padding"] is True


def plate_logits(plate, slots=3, strength=12.0):
    """Build a confident fixture, including the model's trailing padding slots."""
    # Fixture rows preserve the training vocabulary order, including padding.
    rows = [[-strength] * (len(CHARACTERS) + 1) for _ in range(slots)]
    for position, character in enumerate(plate):
        # The positive fixture class is the supplied visible character.
        rows[position][CHARACTERS.index(character) + 1] = strength
    for position in range(len(plate), slots):
        # A short plate must be rewarded for predicting trailing padding.
        rows[position][0] = strength
    return rows


class EvaluationTests(unittest.TestCase):
    """Verify denominators, ranking recall, and calibration on known outcomes."""

    def test_perfect_predictions_include_short_plate_padding(self):
        """Exact recognition and non-padding accuracy use different denominators."""
        # Both short and full-width labels exercise trailing padding handling.
        metrics = evaluate_logits([plate_logits("AB"), plate_logits("C12")], ["AB", "C12"], budgets=(1, 5))
        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["accuracy_exact_plate"], 1.0)
        self.assertEqual(metrics["accuracy_character_nonpadding"], 1.0)
        self.assertEqual(metrics["accuracy_position_including_padding"], 1.0)
        self.assertEqual(metrics["true_plate_recall_at_k"], {"1": 1.0, "5": 1.0})
        self.assertLess(metrics["ece_character_nonpadding"], 1e-6)

    def test_rank_two_plate_is_recalled_without_claiming_top_one_accuracy(self):
        """A useful candidate shortlist can succeed when exact recognition fails."""
        # The first character favors B slightly while A remains the runner-up.
        logits = plate_logits("A", slots=1)
        logits[0][CHARACTERS.index("B") + 1] = 13.0
        metrics = evaluate_logits([logits], ["A"], budgets=(1, 2))
        self.assertEqual(metrics["accuracy_exact_plate"], 0.0)
        self.assertEqual(metrics["true_plate_recall_at_k"], {"1": 0.0, "2": 1.0})
        self.assertEqual(metrics["mean_candidate_count_at_k"], {"1": 1.0, "2": 2.0})

    def test_unreadable_examples_remain_in_recall_denominator(self):
        """Rejecting an ambiguous image must not inflate measured recall."""
        # Equal logits represent missing visual evidence across the whole crop.
        ambiguous = [[0.0] * (len(CHARACTERS) + 1)]
        metrics = evaluate_logits([ambiguous], ["Z"], budgets=(1,))
        self.assertEqual(metrics["sample_count"], 1)
        self.assertEqual(metrics["true_plate_recall_at_k"]["1"], 0.0)
        self.assertEqual(metrics["status_counts"]["unreadable"], 1)

    def test_temperature_scaling_reduces_overconfident_validation_loss(self):
        """Wrong high-confidence decisions should produce a softer fitted scale."""
        # Three repeated logits deliberately disagree with two validation labels.
        logits = [plate_logits("A", slots=1, strength=20.0) for _ in range(3)]
        labels = ["A", "B", "B"]
        temperature = fit_temperature(logits, labels)
        before = evaluate_logits(logits, labels, budgets=(1,), temperature=1.0)
        after = evaluate_logits(logits, labels, budgets=(1,), temperature=temperature)
        self.assertGreater(temperature, 1.0)
        self.assertLess(after["nll_position_including_padding"], before["nll_position_including_padding"])
        self.assertEqual(after["true_plate_recall_at_k"], before["true_plate_recall_at_k"])

    def test_temperature_fit_uses_extreme_finite_logits_without_overflow(self):
        """Stable normalization must tolerate a badly scaled exported checkpoint."""
        # Very large finite logits should not overflow exponentiation.
        logits = [plate_logits("A", slots=1, strength=10000.0)]
        temperature = fit_temperature(logits, ["B"])
        self.assertTrue(math.isfinite(temperature))
        self.assertGreater(temperature, 0.0)

    def test_calibration_predictor_only_receives_validation_records(self):
        """Test labels remain sealed while fitting the confidence temperature."""
        # Distinct path identifiers make accidental cross-split reads observable.
        splits = {
            "train": [SimpleNamespace(path="train.png", plate="A")],
            "val": [SimpleNamespace(path="validation.png", plate="B")],
            "test": [SimpleNamespace(path="test.png", plate="C")],
        }
        visited_paths = []

        def predict(record):
            """Record each access so leakage cannot hide behind correct output."""
            # The callback receives only the row selected by the calibration API.
            visited_paths.append(record.path)
            return plate_logits("A", slots=1)

        fit_validation_temperature(splits, predict)
        self.assertEqual(visited_paths, ["validation.png"])

    def test_calibration_rejects_overlapping_paths(self):
        """A malformed split cannot calibrate using a test image."""
        # The same path under two split names is a direct leakage defect.
        duplicate = SimpleNamespace(path="same.png", plate="A")
        splits = {"train": [], "val": [duplicate], "test": [duplicate]}
        with self.assertRaises(ValueError):
            fit_validation_temperature(splits, lambda record: plate_logits(record.plate))

    def test_invalid_inputs_fail_instead_of_silently_changing_denominators(self):
        """Empty sets, shape mismatch, and impossible labels are caller errors."""
        # Each malformed case would otherwise produce plausible but false metrics.
        invalid_cases = [([], []), ([plate_logits("A")], []), ([plate_logits("A", slots=1)], ["AB"])]
        for logits, labels in invalid_cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_logits(logits, labels)
        with self.assertRaises(ValueError):
            evaluate_logits([plate_logits("A")], ["A"], budgets=(0,))

    def test_latency_units_and_count_are_explicit(self):
        """Measured latency is milliseconds per sample, with one value per row."""
        # Known durations verify the unit and mean independent of host speed.
        metrics = evaluate_logits(
            [plate_logits("A"), plate_logits("B")], ["A", "B"], budgets=(1,), latency_ms=[10.0, 30.0]
        )
        self.assertEqual(metrics["latency_preprocess_model_mean_ms"], 20.0)
        with self.assertRaises(ValueError):
            evaluate_logits([plate_logits("A")], ["A"], latency_ms=[])


class CalibrationTransferTests(unittest.TestCase):
    """Transfer frozen validation scaling without fitting on a target test set."""

    def setUp(self):
        """Keep tiny calibration fixtures independent of torch and model execution."""
        # A private temporary folder owns both the checkpoint bytes and sidecar fixture.
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.checkpoint_path = Path(self.temporary_directory.name) / "recognizer.pt"
        self.checkpoint_path.write_bytes(b"a fixed checkpoint identity")
        self.calibration_path = Path(self.temporary_directory.name) / "calibration.json"
        # These fields match the sidecars emitted by the validation-only CLI.
        self.calibration = {
            "temperature": 1.8,
            "fit_split": "val",
            "validation_sample_count": 20,
            # Fit records bind later test isolation to the actual validation set, not just its name.
            "validation_provenance": [
                {"plate": "CC33CCC", "group": "calibration-session", "path": f"/validation/{index}.png"}
                for index in range(20)
            ],
            "checkpoint_sha256": hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest(),
            "manifest_sha256": "a" * 64,
            "whole_plate_confidence_calibrated": False,
        }

    def write_calibration(self, calibration=None):
        """Write either the valid sidecar or a deliberately malformed replacement."""
        # JSON NaN support is deliberate here so the loader must reject nonfinite temperatures.
        self.calibration_path.write_text(json.dumps(self.calibration if calibration is None else calibration))

    def test_frozen_temperature_loads_without_accessing_or_fitting_test_rows(self):
        """Loading a declared validation artifact does not run the calibration predictor."""
        self.write_calibration()
        with mock.patch("anpr.evaluate.fit_validation_temperature", side_effect=AssertionError("refit")):
            # No records or labels are inputs to frozen transfer; only the checkpoint is identified.
            loaded_calibration = load_calibration(self.calibration_path, self.checkpoint_path)
        self.assertEqual(loaded_calibration, self.calibration)

    def test_calibration_for_another_checkpoint_is_rejected(self):
        """A temperature must stay attached to the exact weights that generated its logits."""
        self.write_calibration()
        # Even a same-path checkpoint replacement must invalidate the original confidence scale.
        self.checkpoint_path.write_bytes(b"different learned weights")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            load_calibration(self.calibration_path, self.checkpoint_path)

    def test_frozen_sidecar_requires_the_actual_validation_records(self):
        """A manifest hash/count cannot reveal which identities were fitted when transferring."""
        # Earlier sidecars omitted these records and must be regenerated rather than guessed.
        legacy_calibration = {
            key: value for key, value in self.calibration.items() if key != "validation_provenance"
        }
        self.write_calibration(legacy_calibration)
        with self.assertRaisesRegex(ValueError, "provenance|regenerate"):
            load_calibration(self.calibration_path, self.checkpoint_path)

    def test_nonvalidation_or_whole_plate_calibration_claims_are_rejected(self):
        """An artifact must not disguise test fitting or assert unsupported plate calibration."""
        # Each changed field contradicts the validation-only positional calibration contract.
        invalid_changes = [
            {"fit_split": "test"},
            {"fit_split": "train"},
            {"whole_plate_confidence_calibrated": True},
            {"whole_plate_confidence_calibrated": 0},
            {"whole_plate_confidence_calibrated": None},
            {"manifest_sha256": "not-a-manifest-hash"},
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                self.write_calibration({**self.calibration, **changes})
                with self.assertRaises(ValueError):
                    load_calibration(self.calibration_path, self.checkpoint_path)

    def test_invalid_temperature_types_and_nonfinite_values_are_rejected(self):
        """Booleans and strings must not silently become plausible numeric temperatures."""
        # A strictly positive finite scalar is required even when JSON accepts NaN or Infinity.
        for temperature in [True, False, "1.8", None, 0, -1, math.nan, math.inf, 10**400]:
            with self.subTest(temperature=temperature):
                self.write_calibration({**self.calibration, "temperature": temperature})
                with self.assertRaisesRegex(ValueError, "temperature"):
                    load_calibration(self.calibration_path, self.checkpoint_path)

    def test_validation_count_must_be_a_positive_integer(self):
        """A declaration with no validation examples cannot justify frozen scaling."""
        # bool is an int subclass, so this check must distinguish explicit sample counts.
        for sample_count in [True, False, 0, -1, 1.5, "20", None]:
            with self.subTest(sample_count=sample_count):
                self.write_calibration({**self.calibration, "validation_sample_count": sample_count})
                with self.assertRaisesRegex(ValueError, "validation_sample_count"):
                    load_calibration(self.calibration_path, self.checkpoint_path)

    def test_sidecar_requires_a_json_object_and_all_calibration_claims(self):
        """Missing provenance or list-shaped JSON must fail rather than default to fitted status."""
        # Required fields must be explicit; false whole-plate calibration cannot be inferred from absence.
        for invalid_document in [
            [],
            "calibration",
            {},
            {
                key: value
                for key, value in self.calibration.items()
                if key != "whole_plate_confidence_calibrated"
            },
        ]:
            with self.subTest(document=invalid_document):
                self.write_calibration(invalid_document)
                with self.assertRaises(ValueError):
                    load_calibration(self.calibration_path, self.checkpoint_path)

    def test_cli_rejects_conflicting_temperature_sources_before_model_loading(self):
        """Users must choose one confidence source rather than silently overriding an artifact."""
        # Required paths are placeholders because argument validation precedes all model/file reads.
        required_arguments = [
            "--checkpoint",
            "unused.pt",
            "--manifest",
            "unused.csv",
            "--output",
            "unused.json",
        ]
        for conflicting_options in [
            ["--calibration", "frozen.json", "--calibrate"],
            ["--calibration", "frozen.json", "--temperature", "2"],
            ["--calibrate", "--temperature", "2"],
        ]:
            with self.subTest(options=conflicting_options), self.assertRaises(SystemExit) as error:
                main(required_arguments + conflicting_options)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    # Running this file directly uses the same assertions as test discovery.
    unittest.main()
