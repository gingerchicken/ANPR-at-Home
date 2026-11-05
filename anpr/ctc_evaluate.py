"""Evaluate CTC plate sequences without treating frames as aligned character heads.

Temperature fitting uses exact target-sequence likelihood on validation crops.
Candidate search is bounded and approximate; retained candidates have exact CTC
scores. Neither those scores nor ordered character marginals are established
probabilities that a registration is correct in the deployment camera domain.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from anpr.alphabet import CHARACTERS, normalize_plate
from anpr.ctc_decoding import ctc_sequence_log_score, decode_ctc_logits
from anpr.evaluate import (
    _expected_calibration_error,
    _file_sha256,
    load_calibration,
    validate_calibration_test_overlap,
    validate_checkpoint_splits,
)
from anpr.paths import protect_output_paths

# A fixed log-spaced grid avoids claiming a convex temperature objective for summed CTC paths.
DEFAULT_TEMPERATURES = tuple(2 ** (-2 + step / 4) for step in range(17))


def _validate_options(budgets, temperature, max_length, beam_width):
    """Validate resource bounds and return reproducibly sorted candidate budgets."""
    # CTC decoding has the same external candidate cap as the independent-head recognizer.
    if not budgets or any(type(budget) is not int or not 1 <= budget <= 1000 for budget in budgets):
        raise ValueError("Candidate budgets must be integers between 1 and 1000")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be a finite positive number")
    if type(max_length) is not int or not 1 <= max_length <= 32:
        raise ValueError("max_length must be an integer between 1 and 32")
    if type(beam_width) is not int or not max(budgets) <= beam_width <= 1000:
        raise ValueError("beam_width must cover every requested budget and be at most 1000")
    return sorted(set(budgets))


def _validated_ctc_samples(logits, labels, max_length):
    """Copy finite frame logits and labels, rejecting targets without a possible alignment."""
    # Every evaluated image needs its own label and at least one sequence frame.
    if not logits or len(logits) != len(labels):
        raise ValueError("Expected nonempty logits and exactly one label per image")
    # Samples may have different frame counts, unlike the fixed positional-head evaluator.
    numeric_logits, normalized_labels = [], []
    for sample_frames, label in zip(logits, labels):
        # A repeated adjacent character needs an intervening CTC blank somewhere in its path.
        normalized_label = normalize_plate(label)
        required_frames = len(normalized_label) + sum(
            first == second for first, second in zip(normalized_label, normalized_label[1:])
        )
        if len(normalized_label) > max_length:
            raise ValueError("CTC target exceeds the configured plate capacity")
        if not required_frames <= len(sample_frames) <= 256:
            raise ValueError("CTC target cannot align to the supplied frame count, or exceeds frame limit")
        # Copy caller-owned scores without mutating tensors or lists used by another report condition.
        converted_frames = []
        for class_scores in sample_frames:
            if len(class_scores) != len(CHARACTERS) + 1:
                raise ValueError("Each CTC frame must contain blank plus the 36 supported characters")
            # Finite raw logits keep exact sequence likelihood and strict JSON metrics meaningful.
            numeric_scores = [float(score) for score in class_scores]
            if not all(math.isfinite(score) for score in numeric_scores):
                raise ValueError("CTC logits must be finite")
            converted_frames.append(numeric_scores)
        numeric_logits.append(converted_frames)
        normalized_labels.append(normalized_label)
    return numeric_logits, normalized_labels


def character_edit_distance(predicted: str, target: str) -> int:
    """Return Levenshtein insertions, deletions and substitutions between two strings.

    Repeated symbols remain distinct characters. Inputs are already normalized
    by evaluation. Rolling rows use O(min(len(predicted), len(target))) memory.
    """
    # Distance is symmetric, so the shorter string can own the rolling row storage.
    if len(predicted) < len(target):
        predicted, target = target, predicted
    # The empty-prefix row represents deleting every target character before consuming a prediction.
    previous_row = list(range(len(target) + 1))
    for predicted_position, predicted_character in enumerate(predicted, start=1):
        # The first column measures insertion cost against an empty target prefix.
        current_row = [predicted_position]
        for target_position, target_character in enumerate(target, start=1):
            # Substitution competes with one insertion or deletion at each alignment cell.
            current_row.append(
                min(
                    previous_row[target_position] + 1,
                    current_row[target_position - 1] + 1,
                    previous_row[target_position - 1] + (predicted_character != target_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def fit_ctc_temperature(logits, labels, *, temperatures=None, max_length=7):
    """Select a bounded-grid scale using mean exact CTC sequence NLL.

    Callers must supply validation samples only. Summing over CTC alignments
    does not justify the positional evaluator's convex bisection. The grid is a
    limited search, not a proof of a global optimum or calibrated correctness.
    """
    # Explicit grid values remain finite and positive before any likelihood calculation.
    requested_grid = DEFAULT_TEMPERATURES if temperatures is None else tuple(temperatures)
    if not requested_grid or any(
        type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0 for scale in requested_grid
    ):
        raise ValueError("Temperature grid must contain finite positive numbers")
    # Reuse capacity validation without invoking approximate beam search during fitting.
    _validate_options((1,), 1.0, max_length, 1)
    numeric_logits, normalized_labels = _validated_ctc_samples(logits, labels, max_length)
    # Grid sorting and a tie preference for one make repeated identical fits reproducible.
    grid = sorted(set(float(scale) for scale in requested_grid))
    scored_temperatures = []
    for scale in grid:
        # Target scores sum all alignments and do not depend on whether a beam retained the true string.
        target_scores = [
            ctc_sequence_log_score(frames, plate, temperature=scale)
            for frames, plate in zip(numeric_logits, normalized_labels)
        ]
        if not all(math.isfinite(score) for score in target_scores):
            raise ValueError("CTC calibration target has no finite alignment likelihood")
        # Mean sequence NLL gives each validation crop equal weight, without padding labels.
        mean_nll = -math.fsum(target_scores) / len(target_scores)
        scored_temperatures.append((mean_nll, abs(math.log(scale)), scale))
    return min(scored_temperatures)[2]


def fit_validation_ctc_temperature(splits, predict, *, temperatures=None, max_length=7):
    """Fit using validation records only, checking identity/source/path isolation first.

    ``predict(record)`` returns that crop's raw frame logits. Split component
    assignment and original-checkpoint isolation are also validated by the CLI.
    """
    # An empty validation source cannot support a fitted temperature declaration.
    validation_records = splits.get("val", [])
    if not validation_records:
        raise ValueError("CTC temperature fitting requires nonempty validation data")
    # Resolve fit paths explicitly so the shared transfer protection uses the same canonical contract.
    fit_provenance = {
        "validation_provenance": [
            {"plate": record.plate, "group": record.group, "path": str(Path(record.path).resolve())}
            for record in validation_records
        ]
    }
    other_records = [record for split, records in splits.items() if split != "val" for record in records]
    validate_calibration_test_overlap(fit_provenance, other_records)
    # Only this validation list enters model execution and the fitted objective.
    validation_logits = [predict(record) for record in validation_records]
    validation_labels = [record.plate for record in validation_records]
    return fit_ctc_temperature(
        validation_logits, validation_labels, temperatures=temperatures, max_length=max_length
    )


def load_ctc_calibration(path, checkpoint_path):
    """Load a frozen CTC scale through shared hash, fit-membership and model-kind checks."""
    # Shared validation rejects mismatched weights and legacy artifacts lacking actual fitted records.
    return load_calibration(path, checkpoint_path, expected_model_kind="ctc")


def evaluate_ctc_logits(
    logits, labels, budgets=(5, 10, 50), temperature=1.0, max_length=7, beam_width=100, latency_ms=None
):
    """Measure complete-plate recall, CER, target CTC NLL and raw top1-score ECE.

    No row is removed for an unreadable status. Character error rate divides
    edit count by true character count and can exceed one when insertions dominate.
    Sequence ECE bins exact CTC likelihood against exact top1 correctness; it is
    a diagnostic and does not certify real-camera confidence. Frame-alignment
    alternatives are not used as aligned character marginal probabilities.
    """
    # Bound search allocation before copying a large batch or calling the decoder.
    candidate_budgets = _validate_options(budgets, temperature, max_length, beam_width)
    numeric_logits, normalized_labels = _validated_ctc_samples(logits, labels, max_length)
    sample_count = len(normalized_labels)
    if latency_ms is not None and (
        len(latency_ms) != sample_count
        or any(
            type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
            for duration in latency_ms
        )
    ):
        raise ValueError("Latency needs one finite nonnegative millisecond duration per sample")
    # Integer counters preserve all denominators, including missing candidates and repeated characters.
    recalled_counts = {budget: 0 for budget in candidate_budgets}
    candidate_counts = {budget: 0 for budget in candidate_budgets}
    exact_count, edit_count = 0, 0
    total_characters = sum(len(label) for label in normalized_labels)
    # Raw sequence confidence and true-sequence NLL remain distinct from per-frame softmax values.
    confidences, correctness, true_plate_scores, decode_durations_ms = [], [], [], []
    status_counts = Counter()
    for frames, label in zip(numeric_logits, normalized_labels):
        # Report decoder cost separately from image loading and neural network execution.
        started = time.perf_counter()
        decoded = decode_ctc_logits(
            frames,
            temperature=temperature,
            top_k=max(candidate_budgets),
            max_length=max_length,
            beam_width=beam_width,
        )
        decode_durations_ms.append((time.perf_counter() - started) * 1000.0)
        candidates = decoded["candidates"]
        candidate_plates = [candidate["plate"] for candidate in candidates]
        best_plate = candidate_plates[0] if candidates else ""
        is_correct = bool(candidates) and best_plate == label
        status_counts[decoded["status"]] += 1
        exact_count += int(is_correct)
        edit_count += character_edit_distance(best_plate, label)
        for budget in candidate_budgets:
            # Candidate lists can be shorter than the request; empty lists are still full test rows.
            recalled_counts[budget] += int(label in candidate_plates[:budget])
            candidate_counts[budget] += min(len(candidate_plates), budget)
        # No renormalization over the short list: confidence uses exact full CTC sequence likelihood.
        best_log_score = candidates[0]["log_score"] if candidates else None
        if best_log_score is not None and (not math.isfinite(best_log_score) or best_log_score > 1e-8):
            raise ValueError("Top CTC sequence log likelihood must be finite and at most zero")
        confidences.append(math.exp(min(0.0, best_log_score)) if candidates else 0.0)
        correctness.append(is_correct)
        # Score the target independently of beam retention so failures still contribute finite NLL.
        target_score = ctc_sequence_log_score(frames, label, temperature=temperature)
        if not math.isfinite(target_score):
            raise ValueError("CTC target cannot align with finite likelihood to the model frames")
        true_plate_scores.append(min(0.0, target_score))
    # Field names identify sequence-level diagnostics and avoid positional-head calibration claims.
    metrics = {
        "sample_count": sample_count,
        "accuracy_exact_plate": exact_count / sample_count,
        "true_plate_recall_at_k": {
            str(budget): recalled_counts[budget] / sample_count for budget in candidate_budgets
        },
        "mean_candidate_count_at_k": {
            str(budget): candidate_counts[budget] / sample_count for budget in candidate_budgets
        },
        "character_error_rate": edit_count / total_characters,
        "character_edit_count": edit_count,
        "target_character_count": total_characters,
        "nll_true_plate_ctc": -math.fsum(true_plate_scores) / sample_count,
        "nll_true_plate_ctc_per_character": -math.fsum(true_plate_scores) / total_characters,
        "ece_top1_sequence_score": _expected_calibration_error(confidences, correctness),
        "ece_bins": 10,
        "confidence_kind": "exact_ctc_sequence_likelihood_not_calibrated_correctness",
        "whole_plate_confidence_calibrated": False,
        "status_counts": dict(status_counts),
        "temperature": temperature,
        "beam_width": beam_width,
        "beam_search_exact": False,
        "retained_candidate_scores_exact_ctc": True,
        "latency_decode_mean_ms": math.fsum(decode_durations_ms) / sample_count,
    }
    if latency_ms is not None:
        # Nearest-rank p95 has a defined result even for a very small held-out set.
        sorted_durations = sorted(latency_ms)
        metrics["latency_preprocess_model_mean_ms"] = math.fsum(latency_ms) / sample_count
        metrics["latency_preprocess_model_p95_ms"] = sorted_durations[math.ceil(0.95 * sample_count) - 1]
    return metrics


def main(argv=None):
    """Evaluate CTC test crops, optionally fitting on a separate validation split."""
    # Argument parsing and destination protection precede optional ML imports and model IO.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 50])
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # A frozen transfer, a fresh validation fit and a manual scale are separate confidence sources.
    confidence_sources = parser.add_mutually_exclusive_group()
    confidence_sources.add_argument("--calibrate", action="store_true")
    confidence_sources.add_argument("--calibration")
    confidence_sources.add_argument("--temperature", type=float)
    parser.add_argument("--calibration-output")
    parser.add_argument("--dataset-provenance")
    parser.add_argument("--data-kind", choices=["real", "synthetic", "unspecified"], default="unspecified")
    parser.add_argument(
        "--degradations",
        nargs="+",
        choices=["clean", "defocus", "motion", "downsample", "jpeg", "crop"],
        default=["clean", "defocus", "motion", "downsample", "jpeg", "crop"],
    )
    parser.add_argument("--severities", type=float, nargs="+", default=[1.0, 2.0])
    arguments = parser.parse_args(argv)
    if arguments.calibration_output and not arguments.calibrate:
        parser.error("--calibration-output requires --calibrate")
    if arguments.threads < 1 or any(
        not math.isfinite(severity) or severity <= 0 for severity in arguments.severities
    ):
        parser.error("Threads and degradation severities must be positive")
    # Validate search bounds before reading a model; actual model capacity is checked after loading.
    _validate_options(
        arguments.budgets,
        arguments.temperature if arguments.temperature is not None else 1.0,
        7,
        arguments.beam_width,
    )
    input_paths = [arguments.checkpoint, arguments.manifest]
    input_paths.extend(path for path in (arguments.calibration, arguments.dataset_provenance) if path)
    output_paths = [arguments.output] + (
        [arguments.calibration_output] if arguments.calibration_output else []
    )
    protect_output_paths(input_paths, output_paths)
    # Pure metric imports remain usable without torch or Pillow installed.
    import torch
    from PIL import Image
    from anpr.data import apply_degradation, load_manifest, split_records
    from anpr.model import load_checkpoint, prepare_image

    # Explicit CPU worker count keeps comparison settings reproducible on the development machine.
    torch.set_num_threads(arguments.threads)
    model, checkpoint_metadata = load_checkpoint(arguments.checkpoint, device=arguments.device)
    if checkpoint_metadata.get("model_kind", "positional") != "ctc":
        raise ValueError("CTC evaluation requires a CTC checkpoint, not positional character heads")
    model.eval()
    records = load_manifest(arguments.manifest, max_length=model.max_length)
    # Image/mask ownership remains protected even when an unused training row references the destination.
    input_paths.extend(record.path for record in records)
    input_paths.extend(record.mask_path for record in records if record.mask_path is not None)
    protect_output_paths(input_paths, output_paths)
    splits = split_records(records, seed=arguments.seed)
    if not splits["test"]:
        raise ValueError("CTC evaluation requires nonempty held-out test data")
    validate_checkpoint_splits(checkpoint_metadata, splits)
    # Source declarations are fingerprinted but do not certify annotation correctness.
    dataset_provenance = None
    if arguments.dataset_provenance:
        provenance_path = Path(arguments.dataset_provenance)
        declarations = json.loads(provenance_path.read_text())
        if not isinstance(declarations, dict):
            raise ValueError("Dataset provenance must contain a JSON object")
        dataset_provenance = {
            "path": str(provenance_path.resolve()),
            "sha256": _file_sha256(provenance_path),
            "declarations": declarations,
        }

    def predict_record(record, degradation="clean", severity=1.0):
        """Time source reading, deterministic corruption, preprocessing and synchronized inference."""
        # CPU conversion synchronizes CUDA before the final timing measurement.
        started = time.perf_counter()
        with Image.open(record.path) as source_image:
            image = apply_degradation(source_image.convert("RGB"), degradation, severity, seed=arguments.seed)
        image_batch = prepare_image(image).unsqueeze(0).to(arguments.device)
        with torch.inference_mode():
            frame_logits = model(image_batch)[0].detach().cpu().tolist()
        return frame_logits, (time.perf_counter() - started) * 1000.0

    # Model checkpoint scaling is a default, not a fitted-confidence claim on the target camera.
    temperature = float(
        arguments.temperature
        if arguments.temperature is not None
        else checkpoint_metadata.get("temperature", 1.0)
    )
    calibration = None
    if arguments.calibration:
        # Frozen scaling never executes the validation predictor or reads target labels for fitting.
        calibration = load_ctc_calibration(arguments.calibration, arguments.checkpoint)
        temperature = calibration["temperature"]
    elif arguments.calibrate:
        # This callback receives validation records only; the grid objective sums exact CTC alignments.
        temperature = fit_validation_ctc_temperature(
            splits, lambda record: predict_record(record)[0], max_length=model.max_length
        )
        calibration = {
            "temperature": temperature,
            "model_kind": "ctc",
            "fit_split": "val",
            "validation_sample_count": len(splits["val"]),
            "validation_provenance": [
                {"plate": record.plate, "group": record.group, "path": str(Path(record.path).resolve())}
                for record in splits["val"]
            ],
            "checkpoint_sha256": _file_sha256(arguments.checkpoint),
            "manifest_sha256": _file_sha256(arguments.manifest),
            "objective": "mean exact CTC true-sequence NLL on original validation crops",
            "temperature_grid": list(DEFAULT_TEMPERATURES),
            "grid_boundary_selected": temperature in (DEFAULT_TEMPERATURES[0], DEFAULT_TEMPERATURES[-1]),
            "whole_plate_confidence_calibrated": False,
        }
    if calibration is not None:
        # Later calibration membership is protected as well as original checkpoint development membership.
        validate_calibration_test_overlap(calibration, splits["test"])
    if arguments.calibration_output:
        # Persist a valid source-bound artifact only after confirming the evaluation's test isolation.
        calibration_path = Path(arguments.calibration_output)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
    # CTC frames carry blank/repeat semantics, so positional-head metrics would be misleading here.
    report = {
        "data_kind": arguments.data_kind,
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "checkpoint_sha256": _file_sha256(arguments.checkpoint),
        "manifest": str(Path(arguments.manifest).resolve()),
        "manifest_sha256": _file_sha256(arguments.manifest),
        "seed": arguments.seed,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "evaluated_split": "test",
        "temperature": temperature,
        "calibration": calibration,
        "model": {
            "model_kind": "ctc",
            "max_length": model.max_length,
            "head_layout": checkpoint_metadata.get("head_layout", "ctc"),
            "preprocess": checkpoint_metadata.get("preprocess"),
        },
        "dataset_provenance": dataset_provenance,
        "decoder": {
            "beam_width": arguments.beam_width,
            "search_exact": False,
            "retained_scores": "exact CTC sequence likelihood summed over all alignments",
            "format_or_lookup_prior": None,
        },
        "limitations": [
            "Prefix beam search is bounded and approximate even though retained scores are exact",
            "Sequence likelihood and ECE do not establish calibrated real-camera correctness",
            "Ordered character marginals are model probabilities, not calibrated character accuracy",
            "Separate alignment-frame snapshots are not character posterior marginals",
            "CER may exceed one and is not the positional-head evaluator's character accuracy",
            "Temperature selection is limited to the declared validation grid, not a global optimum",
            "Source-domain temperature may not transfer to stronger blur or another camera",
            "Source declarations do not independently verify benchmark annotations",
            "Latency includes source reading and preprocessing; decoder latency is reported separately",
        ],
        "results": {},
    }
    for degradation in arguments.degradations:
        # Original crops are scored once; every corruption uses all configured severity settings.
        severities = [0.0] if degradation == "clean" else arguments.severities
        for severity in severities:
            # Conditions share sample order and every test row, including unreadable predictions.
            condition = "clean" if degradation == "clean" else f"{degradation}:{severity:g}"
            print(f"Evaluating CTC {condition} on {len(splits['test'])} test crops", flush=True)
            logits, durations_ms = [], []
            for record in splits["test"]:
                frames, duration_ms = predict_record(record, degradation, max(severity, 1e-9))
                logits.append(frames)
                durations_ms.append(duration_ms)
            report["results"][condition] = evaluate_ctc_logits(
                logits,
                [record.plate for record in splits["test"]],
                arguments.budgets,
                temperature=temperature,
                max_length=model.max_length,
                beam_width=arguments.beam_width,
                latency_ms=durations_ms,
            )
    # Strict JSON prevents a hidden infinite target NLL from masquerading as a completed report.
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved held-out CTC evaluation to {output_path}")


if __name__ == "__main__":
    # Importing metric helpers never starts model loading, calibration or evaluation.
    main()
