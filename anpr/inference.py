"""Recognize standalone plate crops or original-image YOLO detections."""

import argparse
from glob import iglob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from anpr.decoding import decode_logits
from anpr.evaluate import load_calibration
from anpr.model import load_checkpoint, prepare_image
from anpr.paths import protect_output_paths


def _local_detector_source_paths(source):
    """Enumerate local media inputs for ownership checks without decoding their pixels."""
    # Protect the literal spelling too: brackets in a real filename can also look like a glob pattern.
    source_paths = list(
        dict.fromkeys([Path(source), *(Path(path) for path in iglob(source, recursive=True))])
    )
    # Pillow identifies raster suffixes; common video suffixes cover the detector's file inputs.
    media_suffixes = set(Image.registered_extensions()) | {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".mpeg",
        ".mpg",
        ".m4v",
        ".wmv",
        ".ts",
        ".asf",
    }
    for source_path in source_paths:
        if source_path.is_dir():
            # Only media files are input assets; an existing JSONL report in the folder remains replaceable.
            for candidate in source_path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in media_suffixes:
                    yield candidate
        else:
            # A named video/image or file-list input owns its exact path regardless of extension.
            yield source_path


class PlateRecognizer:
    """Reusable crop recognizer retaining per-position alternatives and visual scores."""

    def __init__(self, model: torch.nn.Module, metadata: dict | None = None, device: str = "cpu"):
        """Own a model returning B,L,37 positional or B,T,37 CTC logits, identified by model_kind."""
        # Keep decoding configuration beside the model rather than global notebook state.
        self.model = model.to(device).eval()
        self.device = device
        self.metadata = dict(metadata or {})
        # Sequence blanks and positional padding require different score/decoding semantics.
        self.model_kind = getattr(model, "model_kind", self.metadata.get("model_kind", "positional"))
        if self.model_kind not in ("positional", "ctc"):
            raise ValueError("Unsupported recognition model kind")

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu", *, calibration: str | Path | None = None
    ) -> "PlateRecognizer":
        """Rebuild a recognizer, optionally applying its exact validation sidecar.

        No weights are downloaded and no temperature is fitted on caller crops.
        A sidecar from another checkpoint raises ValueError before recognition.
        """
        # The loader validates vocabulary and preprocessing before running user images.
        model, metadata = load_checkpoint(path, device)
        if calibration is not None:
            # Reuse evaluation's hash/provenance checks so both interfaces use the same scale.
            frozen_calibration = load_calibration(
                calibration, path, expected_model_kind=metadata.get("model_kind", "positional")
            )
            metadata = {
                **metadata,
                "temperature": frozen_calibration["temperature"],
                "temperature_source": "validation_sidecar",
            }
        return cls(model, metadata, device)

    def recognize(self, image: Image.Image, **decode_options) -> dict:
        """Recognize one RGB crop and return JSON-safe ranked alternatives.

        Decode options match decode_logits, including explicit format/lookup constraints.
        Timing covers preprocessing, forward execution, CPU transfer and ranking.
        Status remains uncertain unless the evidence heuristic marks it unreadable.
        """
        # Transfer and CPU list conversion synchronize CUDA before the elapsed time is read.
        started_at = time.perf_counter()
        image_batch = prepare_image(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            # No input requires gradients or updates inference batch statistics.
            logits = self.model(image_batch)[0].detach().cpu().tolist()
        # A validation-fitted temperature can be overridden explicitly for reproducible reports.
        options = {"temperature": self.metadata.get("temperature", 1.0), **decode_options}
        if self.model_kind == "ctc":
            # CTC scores sum complete alignments; frame softmax values cannot be multiplied by character slot.
            from anpr.ctc_decoding import decode_ctc_logits

            # Model capacity limits returned plate length independently of the number of feature timesteps.
            requested_length = options.get("max_length")
            if requested_length is None:
                options["max_length"] = self.model.max_length
            elif isinstance(requested_length, int) and requested_length > self.model.max_length:
                raise ValueError("Requested plate length exceeds the trained CTC model capacity")
            prediction = decode_ctc_logits(logits, **options)
        else:
            # Keep fixed-position ranking exactly as before for all existing checkpoints.
            prediction = decode_logits(logits, **options)
        prediction["latency_ms"] = (time.perf_counter() - started_at) * 1000.0
        prediction["input_size"] = list(image.size)
        # An explicit per-call scale overrides checkpoint or validation settings without mislabelling it.
        prediction["temperature_source"] = (
            "explicit_override"
            if "temperature" in decode_options
            else self.metadata.get("temperature_source", "checkpoint")
        )
        # Split provenance belongs to evaluation, not every public crop result or video frame.
        prediction["model_metadata"] = {
            key: self.metadata[key]
            for key in ("max_length", "use_mask", "head_layout", "model_kind", "alphabet", "preprocess")
            if key in self.metadata
        }
        return prediction


def recognize_yolo_result(
    result,
    recognizer: PlateRecognizer,
    *,
    plate_class_id: int = 0,
    padding_fraction: float = 0.03,
    **decode_options,
) -> list[dict]:
    """Crop an Ultralytics Results object's original BGR image, then run recognition.

    Uses duck typing so standalone OCR needs no ultralytics installation. Bounding
    boxes are pixel xyxy coordinates; detector confidence is separate from OCR
    scores. Boxes outside the image or with no area are skipped. Nonfinite geometry
    fails explicitly. Annotated plots are never used as recognition evidence.
    """
    if not math.isfinite(padding_fraction) or not 0 <= padding_fraction <= 1:
        # An accidental huge margin would feed unrelated vehicle pixels to OCR.
        raise ValueError("padding_fraction must be finite and between zero and one")
    if result.boxes is None:
        # A valid frame without detections has no plate recognition results.
        return []
    # Ultralytics orig_img is BGR HWC; create RGB only once for all crops in the frame.
    original_pixels = np.asarray(result.orig_img)
    if original_pixels.ndim != 3 or original_pixels.shape[2] != 3:
        raise ValueError("Expected an original three-channel BGR detector image")
    original_image = Image.fromarray(original_pixels[:, :, ::-1].copy())
    image_width, image_height = original_image.size
    # Detach to CPU once per frame rather than triggering transfers for every coordinate.
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    predictions = []
    for coordinates, class_id, confidence in zip(boxes, classes, confidences, strict=True):
        if (
            not math.isfinite(class_id)
            or class_id < 0
            or class_id != int(class_id)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            # Invalid detection metadata must not produce misleading or non-JSON output records.
            raise ValueError("Detector class and confidence must be finite and valid")
        # Only the selected detector class represents a registration plate.
        if int(class_id) != plate_class_id:
            continue
        if len(coordinates) != 4 or not all(math.isfinite(number) for number in coordinates):
            raise ValueError("Detector box coordinates must be finite xyxy values")
        # Expand in original-image units and clip before integer slicing.
        left, top, right, bottom = coordinates
        margin_x = max(0, right - left) * padding_fraction
        margin_y = max(0, bottom - top) * padding_fraction
        crop_box = [
            max(0, min(image_width, math.floor(left - margin_x))),
            max(0, min(image_height, math.floor(top - margin_y))),
            max(0, min(image_width, math.ceil(right + margin_x))),
            max(0, min(image_height, math.ceil(bottom + margin_y))),
        ]
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            # Out-of-frame or empty detections cannot supply image evidence.
            continue
        # Recognition reads an independent unannotated RGB crop, preserving its source box.
        prediction = recognizer.recognize(original_image.crop(tuple(crop_box)), **decode_options)
        prediction.update(
            {
                "source": str(result.path),
                "box_xyxy": coordinates,
                "crop_xyxy": crop_box,
                "class_id": int(class_id),
                "detection_confidence": float(confidence),
            }
        )
        predictions.append(prediction)
    return predictions


def main() -> None:
    """Run crop OCR or stream YOLO frames into newline-delimited JSON predictions."""
    # Detector dependencies load only when explicitly requested, keeping crop OCR lightweight.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", help="Standalone plate crop")
    parser.add_argument("--yolo-weights", help="YOLO plate detector checkpoint")
    parser.add_argument("--source", help="YOLO image, video, directory or stream")
    parser.add_argument("--output", help="JSONL destination, otherwise standard output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2, help="CPU PyTorch worker threads")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--beam-width", type=int, help="CTC prefix-search width; approximate ranking only")
    # A scale is either explicitly supplied or bound to checkpoint/validation provenance.
    confidence_source = parser.add_mutually_exclusive_group()
    confidence_source.add_argument("--temperature", type=float)
    confidence_source.add_argument("--calibration", help="Frozen validation sidecar for this checkpoint")
    parser.add_argument("--pattern", help="Explicit L/D/A position mask, for example LLDDLLL")
    parser.add_argument("--known-plates", help="Text file with one known plate per line")
    parser.add_argument("--min-position-confidence", type=float, default=0.0)
    parser.add_argument("--plate-class-id", type=int, default=0)
    parser.add_argument("--detection-confidence", type=float, default=0.25)
    arguments = parser.parse_args()
    if bool(arguments.image) == bool(arguments.yolo_weights) or (
        arguments.yolo_weights and not arguments.source
    ):
        # Separate image-crop mode from detector-source mode to prevent accidental double detection.
        parser.error("Use --image OR --yolo-weights with --source")
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    if arguments.output:
        # Protect image, model, lookup and calibration inputs before the output stream is opened.
        input_paths = [arguments.checkpoint]
        input_paths.extend(
            path
            for path in (
                arguments.image,
                arguments.yolo_weights,
                arguments.calibration,
                arguments.known_plates,
            )
            if path
        )
        if arguments.source and "://" not in arguments.source:
            # A local video is a source asset; remote streams cannot alias a local destination path.
            input_paths.extend(_local_detector_source_paths(arguments.source))
        protect_output_paths(input_paths, [arguments.output])
    # Thread count is explicit because this host has limited CPU and memory resources.
    torch.set_num_threads(arguments.threads)
    recognizer = PlateRecognizer.from_checkpoint(
        arguments.checkpoint, arguments.device, calibration=arguments.calibration
    )
    options = {
        "top_k": arguments.top_k,
        "pattern": arguments.pattern,
        "min_position_confidence": arguments.min_position_confidence,
    }
    if arguments.beam_width is not None:
        # A beam width has no role in the exact independent-head positional decoder.
        if recognizer.model_kind != "ctc":
            parser.error("--beam-width requires a CTC checkpoint")
        options["beam_width"] = arguments.beam_width
    if arguments.temperature is not None:
        # Calibrated validation output is applied only when the caller selects it.
        options["temperature"] = arguments.temperature
    if arguments.known_plates:
        # Blank lines are separators; nonempty labels are validated by the shared decoder.
        options["known_plates"] = [
            line.strip() for line in Path(arguments.known_plates).read_text().splitlines() if line.strip()
        ]
    # An explicit destination is streamed frame by frame rather than held in memory for a video.
    output_file = Path(arguments.output).open("w") if arguments.output else None
    try:

        def emit(prediction: dict) -> None:
            """Write strict JSON so nonfinite scores cannot leak into downstream lookup."""
            # One record per crop allows consumers to handle long videos incrementally.
            serialized_prediction = json.dumps(prediction, allow_nan=False)
            print(serialized_prediction, file=output_file, flush=True)

        if arguments.image:
            # Apply EXIF orientation for standalone images before the crop geometry is normalized.
            with Image.open(arguments.image) as source_image:
                prediction = recognizer.recognize(
                    ImageOps.exif_transpose(source_image).convert("RGB"), **options
                )
            emit({"source": arguments.image, **prediction})
        else:
            # Importing optional detection at this boundary avoids installing YOLO for crop recognition.
            from ultralytics import YOLO

            # Streaming bounds memory even when the source is a long video.
            detector = YOLO(arguments.yolo_weights)
            for frame_index, result in enumerate(
                detector.predict(
                    source=arguments.source, stream=True, conf=arguments.detection_confidence, verbose=False
                )
            ):
                # Frame index keeps repeated video paths distinguishable downstream.
                for prediction in recognize_yolo_result(
                    result, recognizer, plate_class_id=arguments.plate_class_id, **options
                ):
                    emit({"frame_index": frame_index, **prediction})
    finally:
        if output_file is not None:
            # Closing on exceptions leaves a usable prefix of any long-running prediction stream.
            output_file.close()


if __name__ == "__main__":
    # Module execution is explicit; importing this API never starts a detector or opens files.
    main()
