"""Prepare original plate crops from an offline Roboflow YOLO detection export.

Detection annotations contain locations, not registration text. The resulting
CSV deliberately leaves plate and group blank unless an explicit transcription
mapping supplies them. Existing export splits are retained, but training must
still check plate identity and capture-group leakage after labels are completed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path, PurePosixPath

from PIL import Image

from anpr.alphabet import normalize_plate


# Only raster image formats that Pillow can read as original camera frames are scanned.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
# The first six fields form the editable training manifest; remaining fields retain box provenance.
MANIFEST_COLUMNS = [
    "path",
    "plate",
    "group",
    "split",
    "source",
    "box_index",
    "class_id",
    "box_cx",
    "box_cy",
    "box_width",
    "box_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
]


@dataclass(frozen=True)
class _CropPlan:
    """One validated original-image box with an optional exact OCR annotation."""

    # Source image is guaranteed to resolve inside the declared dataset root.
    image_path: Path
    # Row retains portable paths and both normalized and clipped pixel coordinates.
    manifest_row: dict
    # Pixel crop bounds are left/top inclusive and right/bottom exclusive.
    bounds: tuple[int, int, int, int]


def _inside_dataset(path: Path, dataset_path: Path) -> Path:
    """Reject symlink escapes before opening files from an extracted dataset."""
    # Resolve every source independently because files and intermediate directories may be links.
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(dataset_path):
        raise ValueError(f"Dataset source escapes its root: {path}")
    return resolved_path


def _read_annotations(annotations_path: str | Path | None) -> dict[tuple[str, int], tuple[str, str]]:
    """Read explicit source/box transcription mappings without guessing labels."""
    # Missing annotations request a blank template rather than a filename-derived label.
    annotations: dict[tuple[str, int], tuple[str, str]] = {}
    if annotations_path is None:
        return annotations
    with Path(annotations_path).open(newline="", encoding="utf-8-sig") as annotations_file:
        # Header validation catches detector-only CSVs before crops are created.
        reader = csv.DictReader(annotations_file)
        if not {"source", "box_index", "plate", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("Annotations require source, box_index, plate, and group columns")
        for row_number, row in enumerate(reader, start=2):
            # POSIX dataset-relative paths are stable across export locations and host OSes.
            source_text = (row.get("source") or "").strip()
            source_path = PurePosixPath(source_text)
            if (
                not source_text
                or source_path.is_absolute()
                or ".." in source_path.parts
                or "\\" in source_text
                or ":" in source_text
            ):
                raise ValueError(f"Annotation row {row_number} requires a dataset-relative source path")
            # Box index refers to the zero-based original YOLO label line, before class filtering.
            try:
                box_index = int(row.get("box_index") or "")
            except ValueError as error:
                raise ValueError(f"Annotation row {row_number} has an invalid box_index") from error
            if box_index < 0:
                raise ValueError(f"Annotation row {row_number} has a negative box_index")
            # Labels and capture IDs must be supplied together; both blank remain a template.
            plate_text = (row.get("plate") or "").strip()
            group = (row.get("group") or "").strip()
            if bool(plate_text) != bool(group):
                raise ValueError(f"Annotation row {row_number} must provide both plate and group")
            # Normalization removes display separators without truncating longer registrations.
            plate = normalize_plate(plate_text) if plate_text else ""
            key = (source_path.as_posix(), box_index)
            if key in annotations:
                raise ValueError(f"Duplicate annotation mapping for {key}")
            annotations[key] = (plate, group)
    return annotations


def _read_boxes(label_path: Path) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """Validate normalized five-column YOLO boxes, retaining their source line index."""
    # A missing per-image label file represents an unannotated/negative source, as in YOLO exports.
    if not label_path.is_file():
        return []
    # Valid boxes retain raw normalized geometry for later provenance inspection.
    boxes = []
    for box_index, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines()):
        # Empty lines carry no detection; their original line indices remain stable.
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{box_index + 1} requires class cx cy width height")
        try:
            # Class IDs are integer detection categories, never OCR characters.
            class_id = int(fields[0])
            coordinates = tuple(float(field) for field in fields[1:])
        except ValueError as error:
            raise ValueError(f"{label_path}:{box_index + 1} contains invalid numbers") from error
        if class_id < 0 or any(
            not math.isfinite(coordinate) or not 0 <= coordinate <= 1 for coordinate in coordinates
        ):
            raise ValueError(f"{label_path}:{box_index + 1} requires finite normalized boxes and class >= 0")
        if coordinates[2] <= 0 or coordinates[3] <= 0:
            raise ValueError(f"{label_path}:{box_index + 1} requires positive width and height")
        boxes.append((box_index, class_id, coordinates))
    return boxes


def _plan_crops(
    dataset_path: Path, plate_class_id: int, annotations: dict[tuple[str, int], tuple[str, str]]
) -> list[_CropPlan]:
    """Validate sources and mappings before writing any destination artifacts."""
    # The source spelling is retained while valid/val map to the training manifest's val split.
    split_directories = [("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")]
    if (dataset_path / "valid").exists() and (dataset_path / "val").exists():
        raise ValueError("Dataset contains both valid and val directories; choose one validation export")
    # Matched annotation keys expose typoed or wrong-class mappings after source inspection.
    matched_annotations = set()
    plans = []
    for source_split, output_split in split_directories:
        # YOLO export labels mirror image-relative paths, including nested image folders.
        images_path = dataset_path / source_split / "images"
        labels_path = dataset_path / source_split / "labels"
        if not images_path.exists():
            continue
        _inside_dataset(images_path, dataset_path)
        _inside_dataset(labels_path, dataset_path)
        if not images_path.is_dir() or not labels_path.is_dir():
            raise ValueError(f"Split {source_split} requires images and labels directories")
        for image_path in sorted(images_path.rglob("*")):
            # Non-image export metadata is not a recognition input.
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                continue
            _inside_dataset(image_path, dataset_path)
            # Source is an exact portable mapping key, not merely a basename.
            source = image_path.relative_to(dataset_path).as_posix()
            label_path = labels_path / image_path.relative_to(images_path).with_suffix(".txt")
            _inside_dataset(label_path, dataset_path)
            # Invalid annotations in other detector classes also invalidate this source export.
            boxes = _read_boxes(label_path)
            with Image.open(image_path) as original_image:
                # Dimensions define the normalized-to-pixel conversion before any RGB conversion.
                image_width, image_height = original_image.size
            for box_index, class_id, coordinates in boxes:
                if class_id != plate_class_id:
                    # Other detector classes supply no plate crop or guessed OCR label.
                    continue
                # Normalized geometry can extend beyond the image even with legal centers.
                center_x, center_y, box_width, box_height = coordinates
                left = max(0, math.floor((center_x - box_width / 2) * image_width))
                top = max(0, math.floor((center_y - box_height / 2) * image_height))
                right = min(image_width, math.ceil((center_x + box_width / 2) * image_width))
                bottom = min(image_height, math.ceil((center_y + box_height / 2) * image_height))
                if right <= left or bottom <= top:
                    raise ValueError(f"{source} box {box_index} has no pixels after clipping")
                # Hash the full relative source to avoid same-name collisions across splits/folders.
                source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
                crop_path = f"crops/{output_split}/{source_digest}_{box_index:04d}.png"
                annotation_key = (source, box_index)
                plate, group = annotations.get(annotation_key, ("", ""))
                if annotation_key in annotations:
                    # A supplied annotation is consumed only by its exact original source box.
                    matched_annotations.add(annotation_key)
                # Keep original detector geometry separate from the clipped crop extent.
                manifest_row = dict(
                    zip(
                        MANIFEST_COLUMNS,
                        [
                            crop_path,
                            plate,
                            group,
                            output_split,
                            source,
                            box_index,
                            class_id,
                            center_x,
                            center_y,
                            box_width,
                            box_height,
                            left,
                            top,
                            right,
                            bottom,
                        ],
                    )
                )
                plans.append(_CropPlan(image_path, manifest_row, (left, top, right, bottom)))
    if not plans:
        raise ValueError("No matching plate boxes found in train/valid/val/test YOLO export folders")
    if annotations.keys() - matched_annotations:
        # A stale mapping must not appear to have labelled an unrelated crop successfully.
        raise ValueError(
            f"Annotations do not match exported plate boxes: {sorted(annotations.keys() - matched_annotations)}"
        )
    return plans


def prepare_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    plate_class_id: int = 0,
    annotations: str | Path | None = None,
) -> Path:
    """Write lossless RGB crops and a manifest from a local YOLO detection export.

    ``annotations`` may name a CSV with source,box_index,plate,group. Source paths
    are dataset-relative POSIX paths, and box_index is the zero-based original
    label-file line before class filtering. Missing mappings leave both OCR
    fields blank; training's load_manifest rejects them until manually completed.

    No weights, credentials, network calls, filename transcriptions, or rendered
    detector overlays are used. Existing nonempty destinations are rejected.
    Validation precedes destination writes, and a staging directory keeps failed
    image decoding from exposing a half-written manifest. Returns manifest.csv.
    """
    # Canonical roots bound all source reads and destination crop paths.
    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve()
    if not dataset_path.is_dir():
        raise ValueError("dataset must be an extracted YOLO export directory")
    if isinstance(plate_class_id, bool) or not isinstance(plate_class_id, int) or plate_class_id < 0:
        raise ValueError("plate_class_id must be a nonnegative integer")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        # Never overwrite a manifest that may already contain hand-verified OCR labels.
        raise FileExistsError(f"Output exists and is not empty: {output_path}")
    if output_path == dataset_path or dataset_path.is_relative_to(output_path):
        raise ValueError("output must not contain or replace the source dataset")
    for source_split in ("train", "valid", "val", "test"):
        # Outputs nested in source image/label folders would become input on a later run.
        for source_kind in ("images", "labels"):
            source_directory = dataset_path / source_split / source_kind
            if output_path.is_relative_to(source_directory.resolve()):
                raise ValueError("output must be outside source images and labels directories")
    # All coordinates and mappings are validated before creating the output parent.
    transcription_mapping = _read_annotations(annotations)
    crop_plans = _plan_crops(dataset_path, plate_class_id, transcription_mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anpr-prepare-", dir=output_path.parent) as staging_name:
        # Stage beside the destination so the final directory rename stays on one filesystem.
        staging_path = Path(staging_name)
        for image_path, image_plans in groupby(crop_plans, key=lambda crop_plan: crop_plan.image_path):
            # Plans are source-contiguous, so decoding each frame once avoids repeated full-image IO.
            with Image.open(image_path) as original_image:
                # Keep only one RGB frame resident even when an export contains many large images.
                rgb_image = original_image.convert("RGB")
                for crop_plan in image_plans:
                    # Derived paths contain only fixed directories, a digest, and an integer index.
                    destination = staging_path / crop_plan.manifest_row["path"]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Crop original pixels directly; no text, boxes, thresholds, or masks are drawn.
                    crop = rgb_image.crop(crop_plan.bounds)
                    crop.save(destination, format="PNG")
        with (staging_path / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file:
            # Additional provenance columns are ignored safely by the training manifest loader.
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(crop_plan.manifest_row for crop_plan in crop_plans)
        # Rename cannot replace a nonempty directory, protecting outputs from concurrent reruns.
        os.rename(staging_path, output_path)
    return output_path / "manifest.csv"


def main() -> None:
    """Expose offline crop preparation without requiring ML packages or credentials."""
    # CLI fields mirror the API so instructions can be reproduced on an exported dataset.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plate-class-id", default=0, type=int)
    parser.add_argument("--annotations", type=Path)
    arguments = parser.parse_args()
    # Print the concrete artifact and the remaining human transcription requirement.
    manifest_path = prepare_dataset(
        arguments.dataset,
        arguments.output,
        plate_class_id=arguments.plate_class_id,
        annotations=arguments.annotations,
    )
    print(f"Manifest: {manifest_path}")
    print("Fill every plate and group cell before training. Detector boxes do not supply OCR text.")
    print("Export splits are retained; training checks plate identity and capture-group leakage.")


if __name__ == "__main__":
    # Module invocation performs only local dataset preparation.
    main()
