"""Measure plate shortlist recall and calibrate positional scores on validation data.

The pure metric functions require only Python and the decoder. Torch and Pillow
are imported by the CLI so metric regressions can run without an ML installation.
Positional calibration is not a claim of calibrated whole-plate confidence.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from anpr.alphabet import CHARACTERS, encode_plate, normalize_plate
from anpr.decoding import decode_logits
from anpr.paths import protect_output_paths


def _validated_samples(logits, labels):
    """Validate batch alignment and return numeric rows, normalized plates, targets."""
    # Empty or mismatched batches cannot define meaningful recall denominators.
    if not logits or len(logits) != len(labels):
        raise ValueError("Expected nonempty logits with exactly one label per sample")
    # Every image must use the same configured number of positional heads.
    slots = len(logits[0])
    if slots < 1:
        raise ValueError("Logits must contain at least one character position")
    # Numeric rows, normalized labels, and encoded targets remain sample-aligned.
    numeric_logits, normalized_labels, encoded_labels = [], [], []
    for sample_rows, label in zip(logits, labels):
        # Reject malformed rows before decoding so no image silently disappears.
        if len(sample_rows) != slots:
            raise ValueError("All samples must have the same number of positions")
        # Normalization follows the same label rules used by model training.
        normalized_label = normalize_plate(label)
        if not normalized_label or len(normalized_label) > slots:
            raise ValueError("Labels must be nonempty and fit the configured positions")
        # A single image contains one finite score per alphabet class and padding.
        converted_rows = []
        for class_scores in sample_rows:
            # A wrong vocabulary order or width would invalidate every metric.
            if len(class_scores) != len(CHARACTERS) + 1:
                raise ValueError("Each position must contain padding plus A-Z and 0-9")
            # Explicit float conversion supports values from Python and NumPy.
            numeric_scores = [float(score) for score in class_scores]
            if not all(math.isfinite(score) for score in numeric_scores):
                raise ValueError("Logits must be finite")
            converted_rows.append(numeric_scores)
        numeric_logits.append(converted_rows)
        normalized_labels.append(normalized_label)
        encoded_labels.append(encode_plate(normalized_label, slots))
    return numeric_logits, normalized_labels, encoded_labels


def _log_probabilities(class_scores, temperature):
    """Normalize a position without exponentiating large positive logits."""
    # Centering before temperature scaling preserves numerical stability.
    peak_score = max(class_scores)
    centered_scores = [(score - peak_score) / temperature for score in class_scores]
    log_normalizer = math.log(math.fsum(math.exp(score) for score in centered_scores))
    return [score - log_normalizer for score in centered_scores]


def _expected_calibration_error(confidences, correctness, bins=10):
    """Return equal-width ECE; each input represents one evaluated position."""
    # Empty non-padding sets are invalid upstream but remain safe for reuse.
    if not confidences:
        return 0.0
    # Each bin tracks count, summed confidence, and the number of correct guesses.
    bin_counts, bin_confidences, bin_correct = [0] * bins, [0.0] * bins, [0] * bins
    for confidence, is_correct in zip(confidences, correctness):
        # An exact confidence of one belongs in the final bin, not beyond it.
        bin_index = min(int(confidence * bins), bins - 1)
        bin_counts[bin_index] += 1
        bin_confidences[bin_index] += confidence
        bin_correct[bin_index] += int(is_correct)
    # Weight bin discrepancies by their share of evaluated positions.
    weighted_error = 0.0
    for count, confidence_sum, correct_count in zip(bin_counts, bin_confidences, bin_correct):
        if count:
            # Empty bins contribute no evidence and therefore no weight.
            weighted_error += abs(correct_count - confidence_sum) / len(confidences)
    return weighted_error


def fit_temperature(logits, labels, min_temperature=0.05, max_temperature=20.0):
    """Fit scalar temperature by positional validation NLL, including padding.

    The caller must supply validation data only. The inverse-temperature objective
    is convex for fixed logits; bisection of its derivative avoids an optimizer
    dependency. Bounds prevent degenerate fits from tiny validation sets. It costs
    O(50 * samples * positions * classes), with O(samples * positions * classes)
    storage for the validated inputs.
    """
    # Temperature bounds are numerical safeguards, not an accuracy guarantee.
    if not (
        math.isfinite(min_temperature)
        and math.isfinite(max_temperature)
        and 0 < min_temperature < max_temperature
    ):
        raise ValueError("Temperature bounds must be finite, positive, and ordered")
    # Encoded targets include trailing padding to match the training objective.
    numeric_logits, _, targets = _validated_samples(logits, labels)
    # Center each position once because only inverse temperature changes below.
    centered_positions = []
    for sample_rows, target_indices in zip(numeric_logits, targets):
        for class_scores, target_index in zip(sample_rows, target_indices):
            # Adding a constant to logits cannot affect either probabilities or NLL.
            peak_score = max(class_scores)
            centered_positions.append(([score - peak_score for score in class_scores], target_index))

    def derivative(inverse_temperature):
        """Compute mean-NLL derivative up to a positive constant scale."""
        # Its sign suffices for convex bisection; normalization by count is omitted.
        total_derivative = 0.0
        for centered_scores, target_index in centered_positions:
            # All exponent arguments are nonpositive after centering.
            weights = [math.exp(score * inverse_temperature) for score in centered_scores]
            weight_sum = math.fsum(weights)
            expected_score = math.fsum(
                score * (weight / weight_sum) for score, weight in zip(centered_scores, weights) if weight > 0
            )
            total_derivative += expected_score - centered_scores[target_index]
        return total_derivative

    # Inverting the bounds keeps the search variable positive and ordered.
    lower_inverse, upper_inverse = 1.0 / max_temperature, 1.0 / min_temperature
    if derivative(lower_inverse) >= 0:
        # Validation disagreement can place the optimum beyond the softest bound.
        return float(max_temperature)
    if derivative(upper_inverse) <= 0:
        # Consistently correct predictions can prefer a very sharp distribution.
        return float(min_temperature)
    for _ in range(50):
        # The derivative is monotone, so each iteration halves the search interval.
        midpoint_inverse = (lower_inverse + upper_inverse) / 2.0
        if derivative(midpoint_inverse) < 0:
            # Negative slope means increasing inverse temperature still improves NLL.
            lower_inverse = midpoint_inverse
        else:
            # Positive slope means the optimum is at a lower inverse temperature.
            upper_inverse = midpoint_inverse
    return 1.0 / ((lower_inverse + upper_inverse) / 2.0)


def fit_validation_temperature(splits, predict):
    """Fit confidence scaling using only validated validation-split records.

    ``predict(record)`` returns positional logits. Manifest identity/source-group
    leakage must already be checked by ``split_records``. Direct path overlaps are
    also rejected here so a caller cannot calibrate on an obvious test duplicate.
    """
    # Fitting on no examples must fail rather than defaulting to an unexplained T.
    validation_records = splits.get("val", [])
    if not validation_records:
        raise ValueError("Temperature fitting requires nonempty validation data")
    # Paths are a second guard against calibration and test/train data overlap.
    other_paths = {
        str(record.path) for split, records in splits.items() if split != "val" for record in records
    }
    if any(str(record.path) in other_paths for record in validation_records):
        raise ValueError("Validation paths overlap another split")
    # The predictor is deliberately called only for validation rows.
    validation_logits = [predict(record) for record in validation_records]
    validation_labels = [record.plate for record in validation_records]
    return fit_temperature(validation_logits, validation_labels)


def evaluate_logits(logits, labels, budgets=(5, 10, 50), temperature=1.0, latency_ms=None):
    """Measure unconstrained shortlist recall and positional calibration.

    Labels must describe held-out crops in sample order. No unreadable example is
    excluded from accuracy or recall. Candidate scores remain visual model scores;
    the function never labels their products as calibrated plate probabilities.
    Optional timings cover preprocessing plus model execution for each sample.
    """
    # Reject missing/invalid budgets before invoking potentially expensive decoding.
    if not budgets or any(
        isinstance(budget, bool) or not isinstance(budget, int) or budget < 1 for budget in budgets
    ):
        raise ValueError("Candidate budgets must be positive integers")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    # Budget sorting makes JSON reports stable without discarding requested metrics.
    candidate_budgets = sorted(set(budgets))
    numeric_logits, normalized_labels, targets = _validated_samples(logits, labels)
    sample_count = len(normalized_labels)
    if latency_ms is not None and (
        len(latency_ms) != sample_count
        or any(not math.isfinite(duration) or duration < 0 for duration in latency_ms)
    ):
        # A missing timing must not silently reduce its denominator.
        raise ValueError("Latency must contain one finite nonnegative millisecond value per sample")
    # Counts stay in integers until final aggregation to preserve clear denominators.
    recalled_counts = {budget: 0 for budget in candidate_budgets}
    candidate_counts = {budget: 0 for budget in candidate_budgets}
    exact_count, correct_positions, correct_characters = 0, 0, 0
    position_count, character_count = 0, 0
    position_nll, character_nll = 0.0, 0.0
    # Confidence/correctness pairs support separate padded and visible-character ECE.
    position_confidences, position_correctness = [], []
    character_confidences, character_correctness = [], []
    status_counts = Counter()
    decode_durations_ms = []
    for sample_rows, label, target_indices in zip(numeric_logits, normalized_labels, targets):
        # Decoding time is reported separately from image/model execution.
        decode_started = time.perf_counter()
        decoded = decode_logits(sample_rows, temperature=temperature, top_k=max(candidate_budgets))
        decode_durations_ms.append((time.perf_counter() - decode_started) * 1000.0)
        candidate_plates = [candidate["plate"] for candidate in decoded["candidates"]]
        status_counts[decoded["status"]] += 1
        exact_count += int(bool(candidate_plates) and candidate_plates[0] == label)
        for budget in candidate_budgets:
            # All examples count, including unreadable ones and short candidate lists.
            recalled_counts[budget] += int(label in candidate_plates[:budget])
            candidate_counts[budget] += min(len(candidate_plates), budget)
        for class_scores, target_index in zip(sample_rows, target_indices):
            # Argmax positional accuracy is distinct from constrained plate decoding.
            log_probabilities = _log_probabilities(class_scores, temperature)
            predicted_index = max(range(len(class_scores)), key=class_scores.__getitem__)
            confidence = math.exp(log_probabilities[predicted_index])
            is_correct = predicted_index == target_index
            target_nll = -log_probabilities[target_index]
            position_count += 1
            correct_positions += int(is_correct)
            position_nll += target_nll
            position_confidences.append(confidence)
            position_correctness.append(is_correct)
            if target_index != 0:
                # Padding must not inflate reported recognition of visible characters.
                character_count += 1
                correct_characters += int(is_correct)
                character_nll += target_nll
                character_confidences.append(confidence)
                character_correctness.append(is_correct)
    # Report field names state their units and the exact evidence being measured.
    metrics = {
        "sample_count": sample_count,
        "accuracy_exact_plate": exact_count / sample_count,
        "true_plate_recall_at_k": {
            str(budget): recalled_counts[budget] / sample_count for budget in candidate_budgets
        },
        "mean_candidate_count_at_k": {
            str(budget): candidate_counts[budget] / sample_count for budget in candidate_budgets
        },
        "accuracy_character_nonpadding": correct_characters / character_count,
        "accuracy_position_including_padding": correct_positions / position_count,
        "nll_character_nonpadding": character_nll / character_count,
        "nll_position_including_padding": position_nll / position_count,
        "ece_character_nonpadding": _expected_calibration_error(character_confidences, character_correctness),
        "ece_position_including_padding": _expected_calibration_error(
            position_confidences, position_correctness
        ),
        "ece_bins": 10,
        "status_counts": dict(status_counts),
        "latency_decode_mean_ms": math.fsum(decode_durations_ms) / sample_count,
        "temperature": temperature,
        "whole_plate_confidence_calibrated": False,
    }
    if latency_ms is not None:
        # Nearest-rank p95 also works when the held-out set has only a few images.
        ordered_durations = sorted(latency_ms)
        metrics["latency_preprocess_model_mean_ms"] = math.fsum(latency_ms) / sample_count
        metrics["latency_preprocess_model_p95_ms"] = ordered_durations[math.ceil(0.95 * sample_count) - 1]
    return metrics


def _file_sha256(path):
    """Hash inputs in chunks so large model files do not require another full copy."""
    # A content hash identifies the exact checkpoint and manifest used in the report.
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        # Streaming bounds extra memory to one megabyte even for large checkpoints.
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(path, checkpoint_path, *, expected_model_kind=None):
    """Load frozen positional scaling tied to one checkpoint and validation fit.

    This function reads no crop manifest or test labels and never fits a new
    temperature. The sidecar records a validation declaration, not independent
    proof of how its author selected those examples. It raises ValueError for
    malformed calibration claims or a checkpoint identity mismatch.
    """
    # A JSON object preserves the validation provenance produced by --calibrate.
    calibration = json.loads(Path(path).read_text())
    if not isinstance(calibration, dict):
        # A list or scalar cannot identify the fitted temperature and its source.
        raise ValueError("Calibration sidecar must contain a JSON object")
    if expected_model_kind is not None and calibration.get("model_kind", "positional") != expected_model_kind:
        # A scale fitted using aligned-head NLL is not a CTC sequence calibration protocol.
        raise ValueError("Calibration model kind does not match the selected recognition model")
    # Strict scalar checks reject JSON booleans, strings and nonfinite extensions.
    temperature = calibration.get("temperature")
    if (
        type(temperature) not in (int, float)
        or temperature <= 0
        or temperature > sys.float_info.max
        or not math.isfinite(temperature)
    ):
        raise ValueError("Calibration temperature must be a finite positive number")
    # A scale learned from another set of weights is not this checkpoint's calibration.
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if calibration.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Calibration checkpoint hash does not match the selected checkpoint")
    if calibration.get("fit_split") != "val":
        # Test fitting would invalidate held-out confidence measurements.
        raise ValueError("Calibration must declare fit_split='val'")
    # A positive integer count makes missing or nonsensical fit evidence explicit.
    validation_sample_count = calibration.get("validation_sample_count")
    if type(validation_sample_count) is not int or validation_sample_count < 1:
        raise ValueError("Calibration validation_sample_count must be a positive integer")
    # A hash identifies a manifest but cannot identify which of its rows actually fitted this scale.
    fitted_records = _fitted_validation_records(calibration)
    if len(fitted_records) != validation_sample_count:
        raise ValueError("Calibration validation provenance count does not match validation_sample_count")
    # The validation manifest fingerprint stays distinct from the target test manifest.
    manifest_sha256 = calibration.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in manifest_sha256)
    ):
        raise ValueError("Calibration must identify its validation manifest with a SHA256 hash")
    if calibration.get("whole_plate_confidence_calibrated") is not False:
        # Scalar positional temperature does not justify a whole-registration confidence claim.
        raise ValueError("Calibration must explicitly disclaim calibrated whole-plate confidence")
    return calibration


def _fitted_validation_records(calibration):
    """Require explicit fitted identities/sources for future target-split isolation."""
    # Missing legacy records cannot safely be reconstructed from a manifest hash alone.
    fitted_records = calibration.get("validation_provenance")
    if not isinstance(fitted_records, list) or not fitted_records:
        raise ValueError("Calibration lacks fitted validation provenance; regenerate it with --calibrate")
    for record in fitted_records:
        # Sidecars retain past resolved paths without requiring the old files to remain accessible.
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("plate"), str)
            or not isinstance(record.get("group"), str)
            or not record["group"].strip()
            or not isinstance(record.get("path"), str)
            or not Path(record["path"]).is_absolute()
        ):
            raise ValueError(
                "Calibration validation provenance requires plate, source group and absolute path"
            )
        # Unsupported labels must not vanish when building the overlap protection sets.
        normalize_plate(record["plate"])
    return fitted_records


def validate_calibration_test_overlap(calibration, test_records):
    """Reject test identities or sources that fitted a transferred temperature.

    Checkpoint split membership alone misses later calibration on another source.
    These checks apply to benchmark reporting; ordinary inference can still read
    any crop and makes no held-out accuracy claim.
    """
    # Fitted identity and capture sets protect against renamed copies and new crops of the same source.
    fitted_records = _fitted_validation_records(calibration)
    fitted_plates = {normalize_plate(record["plate"]) for record in fitted_records}
    fitted_groups = {record["group"].strip() for record in fitted_records}
    fitted_paths = {str(Path(record["path"]).resolve()) for record in fitted_records}
    for record in test_records:
        # A shared registration, capture group or resolved file disqualifies that test row.
        if (
            normalize_plate(record.plate) in fitted_plates
            or record.group in fitted_groups
            or str(Path(record.path).resolve()) in fitted_paths
        ):
            raise ValueError("Calibration/test leakage: target test overlaps fitted validation records")


def validate_checkpoint_splits(checkpoint_metadata, splits):
    """Reject evaluation/calibration rows exposed during checkpoint training or selection.

    Checking only the new manifest's internal split isolation is insufficient:
    changing its split seed can move the checkpoint's training rows into test.
    Versioned training checkpoints therefore record their original membership.
    """
    # Missing provenance prevents an honest claim that these examples were held out.
    provenance = checkpoint_metadata.get("split_provenance")
    if not isinstance(provenance, dict) or not provenance.get("train") or not provenance.get("val"):
        raise ValueError(
            "Checkpoint lacks train/validation split provenance; retrain before held-out evaluation"
        )
    for target_split, excluded_splits in (("val", ("train",)), ("test", ("train", "val"))):
        # Selection on validation data also disqualifies those identities from final test claims.
        excluded_rows = [row for split in excluded_splits for row in provenance.get(split, [])]
        excluded_plates = {normalize_plate(row["plate"]) for row in excluded_rows}
        excluded_groups = {row["group"] for row in excluded_rows}
        excluded_paths = {str(Path(row["path"]).resolve()) for row in excluded_rows}
        for record in splits.get(target_split, []):
            # Identity, source and resolved file path are independent guards against reuse.
            if (
                normalize_plate(record.plate) in excluded_plates
                or record.group in excluded_groups
                or str(Path(record.path).resolve()) in excluded_paths
            ):
                raise ValueError(
                    f"Checkpoint split leakage: {target_split} overlaps original {excluded_splits}"
                )


def positional_label_coverage(training_rows, test_labels, max_length):
    """Count target classes without positive training examples at the same position.

    This is a label-coverage diagnostic, not recognition accuracy or proof that
    a prediction is impossible. Padding counts as a class because shorter plates
    also need supervised length behavior. Test labels never update model weights.
    """
    # Independent heads need class exposure at their own positions, not just somewhere in a plate.
    observed_classes = [set() for _ in range(max_length)]
    for row in training_rows:
        # Encode with the checkpoint capacity rather than truncating an incompatible training label.
        for position, target_index in enumerate(encode_plate(row["plate"], max_length)):
            observed_classes[position].add(target_index)
    # Counts describe the entire evaluated population, including examples the model cannot read.
    unseen_counts = [0] * max_length
    affected_plates = 0
    for plate in test_labels:
        # All positions, including trailing padding, participate in valid complete-sequence scoring.
        unseen_positions = [
            position
            for position, target_index in enumerate(encode_plate(plate, max_length))
            if target_index not in observed_classes[position]
        ]
        affected_plates += bool(unseen_positions)
        for position in unseen_positions:
            unseen_counts[position] += 1
    return {
        "training_class_count_by_position": [len(classes) for classes in observed_classes],
        "test_plates_with_unseen_position_class": affected_plates,
        "test_unseen_class_count_by_position": unseen_counts,
        "test_sample_count": len(test_labels),
        "includes_trailing_padding": True,
        "interpretation": "Missing positive label exposure is not an accuracy or impossibility claim",
    }


def main(argv=None):
    """Run test-only evaluation, optionally fitting temperature on validation data."""
    # Parsing has no ML side effects, so --help works before dependency installation.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 50])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    # One explicit temperature source prevents accidental refitting while transferring to real test crops.
    confidence_sources = parser.add_mutually_exclusive_group()
    confidence_sources.add_argument("--calibrate", action="store_true")
    confidence_sources.add_argument("--calibration", help="Frozen validation sidecar for this checkpoint")
    confidence_sources.add_argument("--temperature", type=float)
    parser.add_argument("--calibration-output")
    parser.add_argument("--dataset-provenance", help="JSON source/subset declarations for the evaluated data")
    parser.add_argument(
        "--degradations",
        nargs="+",
        default=["clean", "defocus", "motion", "downsample", "jpeg", "crop"],
        choices=["clean", "defocus", "motion", "downsample", "jpeg", "crop"],
    )
    parser.add_argument("--severities", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--data-kind", choices=["real", "synthetic", "unspecified"], default="unspecified")
    # Explicit provenance prevents a synthetic smoke report being mistaken for real evidence.
    arguments = parser.parse_args(argv)
    if arguments.calibration_output and not arguments.calibrate:
        parser.error("--calibration-output requires --calibrate")
    if arguments.threads < 1 or any(
        not math.isfinite(severity) or severity <= 0 for severity in arguments.severities
    ):
        parser.error("Threads and degradation severities must be positive")
    # Reports and fitted scales must never replace model weights or any declared source artifact.
    input_paths = [arguments.checkpoint, arguments.manifest]
    input_paths.extend(path for path in (arguments.calibration, arguments.dataset_provenance) if path)
    output_paths = [arguments.output] + (
        [arguments.calibration_output] if arguments.calibration_output else []
    )
    protect_output_paths(input_paths, output_paths)
    # Optional imports are restricted to actual image/model evaluation.
    import torch
    from PIL import Image
    from anpr.data import apply_degradation, load_manifest, split_records
    from anpr.model import load_checkpoint, prepare_image

    # Small CPU thread counts avoid oversubscribing development machines.
    torch.set_num_threads(arguments.threads)
    model, checkpoint_metadata = load_checkpoint(arguments.checkpoint, device=arguments.device)
    if checkpoint_metadata.get("model_kind", "positional") != "positional":
        # CTC timesteps are not character positions and require sequence-aware metrics and calibration.
        raise ValueError("Use python -m anpr.ctc_evaluate for CTC checkpoints")
    model.eval()
    records = load_manifest(arguments.manifest, max_length=model.max_length)
    # Manifest image/mask files also belong to the dataset even when their split is not evaluated here.
    input_paths.extend(record.path for record in records)
    input_paths.extend(record.mask_path for record in records if record.mask_path is not None)
    protect_output_paths(input_paths, output_paths)
    splits = split_records(records, seed=arguments.seed)
    if not splits["test"]:
        raise ValueError("Evaluation requires nonempty held-out test data")
    # Validate against the checkpoint's actual training split, not merely this invocation's seed.
    validate_checkpoint_splits(checkpoint_metadata, splits)
    # Optional source declarations stay attached to their exact file content, without certifying labels.
    dataset_provenance = None
    if arguments.dataset_provenance:
        # Only object-shaped metadata can describe the source corpus and exported subset coherently.
        provenance_path = Path(arguments.dataset_provenance)
        provenance_content = json.loads(provenance_path.read_text())
        if not isinstance(provenance_content, dict):
            raise ValueError("Dataset provenance must contain a JSON object")
        dataset_provenance = {
            "path": str(provenance_path.resolve()),
            "sha256": _file_sha256(provenance_path),
            "declarations": provenance_content,
        }

    def predict_record(record, degradation="clean", severity=1.0):
        """Read one original crop, apply reproducible degradation, and run inference."""
        # Preparation is timed with inference because both contribute to crop latency.
        started = time.perf_counter()
        with Image.open(record.path) as source_image:
            # Converting inside the context detaches RGB pixels before closing the file.
            image = apply_degradation(
                source_image.convert("RGB"), degradation, severity=severity, seed=arguments.seed
            )
        image_tensor = prepare_image(image).unsqueeze(0).to(arguments.device)
        with torch.inference_mode():
            # CPU transfer synchronizes CUDA before the elapsed time is recorded.
            prediction = model(image_tensor)[0].detach().cpu().tolist()
        duration_ms = (time.perf_counter() - started) * 1000.0
        return prediction, duration_ms

    # Existing checkpoint temperature is used unless explicitly overridden or refitted.
    temperature = float(
        arguments.temperature
        if arguments.temperature is not None
        else checkpoint_metadata.get("temperature", 1.0)
    )
    calibration = None
    if arguments.calibration:
        # Frozen transfer reads no target labels or validation examples and never invokes fitting.
        calibration = load_calibration(
            arguments.calibration, arguments.checkpoint, expected_model_kind="positional"
        )
        temperature = calibration["temperature"]
    elif arguments.calibrate:
        # This API owns validation selection; test rows cannot enter the fit callback.
        temperature = fit_validation_temperature(splits, lambda record: predict_record(record)[0])
        calibration = {
            "temperature": temperature,
            "model_kind": "positional",
            "fit_split": "val",
            "validation_sample_count": len(splits["val"]),
            # These are the actual selected fit rows, including calibration on a later independent source.
            "validation_provenance": [
                {"plate": record.plate, "group": record.group, "path": str(record.path)}
                for record in splits["val"]
            ],
            "objective": "positional NLL including trailing padding on original validation crops",
            "checkpoint_sha256": _file_sha256(arguments.checkpoint),
            "manifest_sha256": _file_sha256(arguments.manifest),
            "whole_plate_confidence_calibrated": False,
        }
        if arguments.calibration_output:
            # A sidecar leaves the selected training checkpoint unchanged and traceable.
            calibration_path = Path(arguments.calibration_output)
            calibration_path.parent.mkdir(parents=True, exist_ok=True)
            calibration_path.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    if calibration is not None:
        # Frozen scaling must not turn its own validation examples into reported held-out test evidence.
        validate_calibration_test_overlap(calibration, splits["test"])
    # The report captures exact artifacts, split allocation, and confidence limitations.
    report = {
        "data_kind": arguments.data_kind,
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "checkpoint_sha256": _file_sha256(arguments.checkpoint),
        "manifest": str(Path(arguments.manifest).resolve()),
        "manifest_sha256": _file_sha256(arguments.manifest),
        "seed": arguments.seed,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "evaluated_split": "test",
        "temperature": temperature,
        "calibration": calibration,
        # Separate label exposure from measured visual recognition when comparing formats or domains.
        "positional_label_coverage": positional_label_coverage(
            checkpoint_metadata["split_provenance"]["train"],
            [record.plate for record in splits["test"]],
            model.max_length,
        ),
        "model": {
            "max_length": checkpoint_metadata.get("max_length", model.max_length),
            "use_mask": checkpoint_metadata.get("use_mask", False),
            "head_layout": checkpoint_metadata.get("head_layout", "global"),
            "preprocess": checkpoint_metadata.get("preprocess"),
        },
        "dataset_provenance": dataset_provenance,
        "decoder_constraints": "trailing padding only; no country-format or known-plate prior",
        "limitations": [
            "Positional ECE and NLL do not establish whole-plate confidence calibration",
            "Synthetic degradation of a crop is not a substitute for naturally blurred real captures",
            "Temperature fitted on original validation crops may not transfer to stronger corruption",
            "Transferred validation calibration does not establish calibration on a different camera domain",
            "Dataset provenance records source declarations, not independent verification of annotation accuracy",
            "Latency includes image reading and preprocessing; decoder timing is reported separately",
        ],
        "results": {},
    }
    for degradation in arguments.degradations:
        # Clean inputs are evaluated once; corruptions retain each requested severity.
        severities = [0.0] if degradation == "clean" else arguments.severities
        for severity in severities:
            # Sample order is fixed across conditions to permit a paired comparison.
            predicted_logits, durations_ms = [], []
            for record in splits["test"]:
                prediction, duration_ms = predict_record(record, degradation, max(severity, 1e-9))
                predicted_logits.append(prediction)
                durations_ms.append(duration_ms)
            condition_name = "clean" if degradation == "clean" else f"{degradation}:{severity:g}"
            report["results"][condition_name] = evaluate_logits(
                predicted_logits,
                [record.plate for record in splits["test"]],
                budgets=arguments.budgets,
                temperature=temperature,
                latency_ms=durations_ms,
            )
    # Strict JSON rejects nonfinite output instead of writing an invalid metrics file.
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved held-out evaluation to {output_path}")


if __name__ == "__main__":
    # Module execution shares the same CLI used in reproducible runbook commands.
    main()
