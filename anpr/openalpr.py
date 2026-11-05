"""Prepare a held-out real-image benchmark from local OpenALPR EU source files.

The upstream annotations use image, x, y, width, height and plate text as six
tab-separated fields. They are benchmark-provided labels, not newly verified
human transcriptions. No detector outputs or filename guesses become labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from PIL import Image

from anpr.alphabet import normalize_plate


# Extra source columns remain auditable while the shared manifest loader ignores them.
MANIFEST_FIELDS = [
    "path",
    "plate",
    "group",
    "split",
    "source",
    "annotation",
    "annotation_row",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
]


def _confined_source(path: Path, source_root: Path) -> Path:
    """Resolve a source path while rejecting symlink escapes from the benchmark."""
    # Both annotation files and referenced image files must remain inside the declared source.
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(source_root):
        raise ValueError(f"Source path escapes outside benchmark folder: {path}")
    return resolved_path


def _sha256(path: Path) -> str:
    """Hash source bytes without keeping a second complete image in memory."""
    # A chunked digest identifies the exact source annotation or image used by this import.
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_openalpr(source: str | Path, output: str | Path, *, max_length: int = 7) -> Path:
    """Create original RGB crops and an entirely held-out OCR manifest.

    Output must be empty. Invalid geometry, malformed annotation rows and unsafe
    source paths fail before destination writes. Unsupported labels and labels
    exceeding model capacity are excluded explicitly in provenance.json, never
    truncated. All accepted rows use split=test and group=openalpr-eu so this
    source cannot silently mix into training. Returns the generated manifest.

    This function performs no download or model inference. If present, LICENSE
    and UPSTREAM.json are copied unchanged, and every consumed source is hashed.
    """
    # Canonical roots define the bounds of all reads and the final destination rename.
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if not source_root.is_dir():
        raise ValueError("source must be a local OpenALPR EU benchmark folder")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= 32:
        raise ValueError("max_length must be an integer from 1 through 32")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        # Label corrections and prior benchmark artifacts must survive accidental reruns.
        raise FileExistsError(f"Output exists and is not empty: {output_root}")
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Output must be separate from the source benchmark folder")
    # Plans contain only validated file paths and geometry, keeping image memory bounded.
    crop_plans = []
    exclusions = []
    source_hashes = {}
    annotation_files = sorted(source_root.glob("*.txt"))
    if not annotation_files:
        raise ValueError("No OpenALPR TXT annotations found")
    for annotation_path in annotation_files:
        # A label symlink cannot make an unrelated file part of this benchmark silently.
        _confined_source(annotation_path, source_root)
        source_hashes[annotation_path.name] = _sha256(annotation_path)
        with annotation_path.open(newline="", encoding="utf-8-sig") as annotation_file:
            # Reading tab fields preserves spaces within a registration for normalization.
            reader = csv.reader(annotation_file, delimiter="\t")
            for row_number, fields in enumerate(reader, start=1):
                if not fields:
                    continue
                if len(fields) != 6:
                    raise ValueError(f"{annotation_path.name}:{row_number} requires six tab-separated fields")
                # Source text is a portable relative image path, never an OCR target guess.
                image_name = fields[0].strip()
                relative_image = PurePosixPath(image_name)
                if (
                    not image_name
                    or relative_image.is_absolute()
                    or ".." in relative_image.parts
                    or "\\" in image_name
                    or ":" in image_name
                ):
                    raise ValueError("Annotation image must be a relative path inside the benchmark")
                image_path = _confined_source(source_root / image_name, source_root)
                if not image_path.is_file():
                    raise ValueError(f"Missing benchmark image: {image_name}")
                # Coordinates use original-image pixels and positive rectangle extents.
                try:
                    left, top, width, height = [float(field) for field in fields[1:5]]
                except ValueError as error:
                    raise ValueError(f"Invalid coordinates in {annotation_path.name}:{row_number}") from error
                if not all(math.isfinite(number) for number in (left, top, width, height)):
                    raise ValueError("Benchmark coordinates must be finite")
                if (
                    width <= 0
                    or height <= 0
                    or not math.isfinite(left + width)
                    or not math.isfinite(top + height)
                ):
                    raise ValueError("Benchmark box width and height must define finite positive extents")
                with Image.open(image_path) as original_image:
                    # Image dimensions validate clipping before any crop files are written.
                    image_width, image_height = original_image.size
                bounds = (
                    max(0, min(image_width, math.floor(left))),
                    max(0, min(image_height, math.floor(top))),
                    max(0, min(image_width, math.ceil(left + width))),
                    max(0, min(image_height, math.ceil(top + height))),
                )
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    raise ValueError("Benchmark box has no pixel area after clipping")
                # Raw labels and their location survive capacity/alphabet exclusions unchanged.
                raw_plate = fields[5]
                source_record = {
                    "source": image_name,
                    "annotation": annotation_path.name,
                    "annotation_row": row_number,
                    "raw_plate": raw_plate,
                    "source_xywh": [left, top, width, height],
                }
                try:
                    plate = normalize_plate(raw_plate)
                except ValueError:
                    # Unsupported text cannot be repaired by deleting unknown characters.
                    exclusions.append({**source_record, "reason": "unsupported_label"})
                    continue
                if len(plate) > max_length:
                    # A model capacity filter changes the denominator and must be reported explicitly.
                    exclusions.append(
                        {
                            **source_record,
                            "reason": "exceeds_model_capacity",
                            "normalized_plate": plate,
                            "length": len(plate),
                        }
                    )
                    continue
                # Hash exact file bytes so later changes to a label or source image are detectable.
                if image_name not in source_hashes:
                    source_hashes[image_name] = _sha256(image_path)
                crop_digest = hashlib.sha256(
                    f"{annotation_path.name}:{row_number}:{image_name}".encode()
                ).hexdigest()
                manifest_row = dict(
                    zip(
                        MANIFEST_FIELDS,
                        [
                            f"crops/{crop_digest}.png",
                            plate,
                            "openalpr-eu",
                            "test",
                            image_name,
                            annotation_path.name,
                            row_number,
                            *bounds,
                        ],
                    )
                )
                crop_plans.append((image_path, bounds, manifest_row))
    # Optional upstream files retain attribution and pin metadata without inventing image provenance.
    upstream_files = [
        source_root / filename
        for filename in ("LICENSE", "UPSTREAM.json")
        if (source_root / filename).exists()
    ]
    for upstream_path in upstream_files:
        _confined_source(upstream_path, source_root)
        source_hashes[upstream_path.name] = _sha256(upstream_path)
    # The report states filtering and unresolved source-quality limits alongside its denominator.
    provenance = {
        "source": str(source_root),
        "source_kind": "real_openalpr_eu_benchmark",
        "label_origin": "upstream_benchmark_annotation",
        "all_rows_split": "test",
        "all_rows_group": "openalpr-eu",
        "max_length": max_length,
        "included_count": len(crop_plans),
        "excluded_count": len(exclusions),
        "exclusions": exclusions,
        "source_sha256": source_hashes,
        "limitations": [
            "Source annotations have not been independently verified as human ground truth",
            "No source-session metadata is supplied; the entire benchmark is held out together",
            "Ground-truth box crops do not demonstrate previously trained YOLO detector performance",
            "Capacity/alphabet exclusions change the evaluated population and must remain visible",
        ],
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".anpr-openalpr-", dir=output_root.parent) as staging_name:
        # Stage complete artifacts beside the destination for one final directory rename.
        staging_root = Path(staging_name)
        for image_path, bounds, manifest_row in crop_plans:
            with Image.open(image_path) as original_image:
                # Only original RGB pixels enter recognition, without annotated box overlays.
                crop = original_image.convert("RGB").crop(bounds)
                destination = staging_root / manifest_row["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                crop.save(destination, format="PNG")
        with (staging_root / "manifest.csv").open("w", newline="", encoding="utf-8") as manifest_file:
            # Additional provenance columns remain compatible with the shared OCR manifest loader.
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest_row for _, _, manifest_row in crop_plans)
        for upstream_path in upstream_files:
            shutil.copyfile(upstream_path, staging_root / upstream_path.name)
        (staging_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n"
        )
        # Existing nonempty output protection also applies if a concurrent import races this rename.
        os.rename(staging_root, output_root)
    return output_root / "manifest.csv"


def main() -> None:
    """Prepare an already-downloaded benchmark without credentials or network access."""
    # The CLI keeps capacity filtering explicit and requires a separate empty output folder.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=7)
    arguments = parser.parse_args()
    try:
        print(prepare_openalpr(arguments.source, arguments.output, max_length=arguments.max_length))
    except (ValueError, OSError) as error:
        parser.exit(2, f"OpenALPR preparation failed: {error}\n")


if __name__ == "__main__":
    # Importing this module never downloads benchmark images or creates crop artifacts.
    main()
