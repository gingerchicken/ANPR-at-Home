"""Load labelled plate crops, prevent split leakage, and generate smoke data.

The foreground masks describe clean character geometry. Image-only blur and
compression deliberately leave those targets unchanged; geometric changes move
the image and mask together. Synthetic results are development checks, not proof
of performance on real camera crops.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    __version__ as pillow_version,
    features,
)

from anpr.alphabet import CHARACTERS, encode_plate, normalize_plate


@dataclass(frozen=True)
class PlateRecord:
    """One labelled crop and its capture provenance; masks are optional targets."""

    # Paths resolve against the manifest directory, never the process directory.
    path: Path
    # Plate is normalized uppercase alphanumeric text, without visual spacing.
    plate: str
    # Group identifies a capture/video/vehicle source that must not span splits.
    group: str
    # Split is absent for automatic assignment or one of train, val, and test.
    split: str | None = None
    # Mask is a same-size grayscale glyph target, white foreground on black.
    mask_path: Path | None = None


def load_manifest(manifest_path: str | Path, max_length: int = 7) -> list[PlateRecord]:
    """Read and validate CSV crops before training; reject missing/invalid labels.

    Required columns are path, plate, and group. Optional split values must be
    supplied for every row or none; split_records enforces their isolation.
    """
    # Resolve image locations consistently even when invoked from another folder.
    manifest_path = Path(manifest_path).resolve()
    # Records hold validated labels and paths; no image content is loaded here.
    records: list[PlateRecord] = []
    # Reusing one resolved file with conflicting labels creates invalid supervision.
    labels_by_path: dict[Path, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        # DictReader retains explicit column names and quoted plate text.
        reader = csv.DictReader(manifest_file)
        if not {"path", "plate", "group"}.issubset(reader.fieldnames or []):
            # Missing provenance cannot be silently replaced with a random row split.
            raise ValueError("Manifest requires path, plate, and group columns")
        for row_number, row in enumerate(reader, start=2):
            # Normalize the label before checking the configured head count.
            plate = normalize_plate(row.get("plate") or "")
            encode_plate(plate, max_length)
            if not plate:
                # Empty crops need a separate rejection model, not padding-only training.
                raise ValueError(f"Manifest row {row_number} has an empty plate")
            # Capture identifiers are mandatory so real video frames stay together.
            group = (row.get("group") or "").strip()
            if not group:
                raise ValueError(f"Manifest row {row_number} has no source group")
            # Relative paths are interpreted relative to this manifest file.
            path = (manifest_path.parent / (row.get("path") or "")).resolve()
            # An optional clean mask supports segmentation supervision later.
            mask_text = (row.get("mask_path") or "").strip()
            mask_path = (manifest_path.parent / mask_text).resolve() if mask_text else None
            if not path.is_file() or (mask_path is not None and not mask_path.is_file()):
                # Fail before allocating a model or downloading pretrained weights.
                raise ValueError(f"Manifest row {row_number} references a missing image or mask")
            if path in labels_by_path and labels_by_path[path] != plate:
                # Do not let a mislabeled duplicate masquerade as an independent sample.
                raise ValueError(f"Manifest row {row_number} assigns conflicting labels to {path}")
            labels_by_path[path] = plate
            # Empty split cells request automatic assignment for the entire manifest.
            split = (row.get("split") or "").strip().lower() or None
            if split is not None and split not in {"train", "val", "test"}:
                raise ValueError(f"Manifest row {row_number} has invalid split {split!r}")
            records.append(PlateRecord(path, plate, group, split, mask_path))
    if not records:
        # An empty manifest otherwise produces misleading zero-loss training.
        raise ValueError("Manifest contains no labelled plate crops")
    return records


def split_records(records: list[PlateRecord], seed: int = 42) -> dict[str, list[PlateRecord]]:
    """Keep transitive plate/source components within seeded 70/15/15 splits.

    Connected components prevent a source containing several plates from linking
    the same identity across train and test. Exact ratios are impossible for
    large components; leakage isolation takes priority. Complexity is O(N alpha(N))
    for union/find plus O(C log C) to assign C components by size.
    """
    # Fixed names and target weights keep assignment stable across Python runs.
    split_names = ("train", "val", "test")
    target_weights = {"train": 0.70, "val": 0.15, "test": 0.15}
    # Parent links represent transitive identity and source relationships.
    parents = list(range(len(records)))
    # Component sizes keep union depth shallow before path compression occurs.
    component_sizes = [1] * len(records)

    def find(index: int) -> int:
        """Find a component root while compressing its lookup path."""
        # Compression prevents long capture/identity chains from costing O(N) per row.
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # First occurrence maps join every plate identity and capture group.
    plate_owner: dict[str, int] = {}
    group_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        # Both relationships matter; using plate-only or source-only misses leakage.
        for owner_map, key in ((plate_owner, normalize_plate(record.plate)), (group_owner, record.group)):
            if key in owner_map:
                # Join the smaller tree below the larger to retain inverse-Ackermann cost.
                first_root, second_root = find(index), find(owner_map[key])
                if first_root != second_root:
                    if component_sizes[first_root] < component_sizes[second_root]:
                        first_root, second_root = second_root, first_root
                    parents[second_root] = first_root
                    component_sizes[first_root] += component_sizes[second_root]
            else:
                owner_map[key] = index
    # Components collect all rows that must move together.
    components: dict[int, list[PlateRecord]] = {}
    for index, record in enumerate(records):
        components.setdefault(find(index), []).append(record)
    # Partitions are returned even when too few independent sources fill all three.
    partitions: dict[str, list[PlateRecord]] = {name: [] for name in split_names}
    # A partial explicit split is ambiguous and risks accidental contamination.
    has_explicit_splits = any(record.split is not None for record in records)
    if has_explicit_splits:
        if any(record.split is None for record in records):
            raise ValueError("Provide a split for every manifest row or leave all splits empty")
        for component in components.values():
            # One connected capture component cannot have multiple assigned splits.
            assigned_splits = {record.split for record in component}
            if len(assigned_splits) != 1:
                raise ValueError("Split leakage: shared plate identity or source appears in multiple splits")
            # The singleton was validated when reading the manifest or below.
            assigned_split = next(iter(assigned_splits))
            if assigned_split not in partitions:
                raise ValueError(f"Invalid split {assigned_split!r}")
            partitions[assigned_split].extend(component)
        return partitions
    # Sorting before seeded shuffling makes assignment independent of CSV row order.
    ordered_components = sorted(
        components.values(), key=lambda component: min(str(record.path) for record in component)
    )
    # Private RNG avoids perturbing augmentation or model initialization randomness.
    generator = random.Random(seed)
    generator.shuffle(ordered_components)
    ordered_components.sort(key=len, reverse=True)
    for component in ordered_components:
        # Normalize current occupancy by its target weight, giving each split a turn.
        assigned_split = min(split_names, key=lambda name: len(partitions[name]) / target_weights[name])
        partitions[assigned_split].extend(replace(record, split=assigned_split) for record in component)
    return partitions


def apply_degradation(image: Image.Image, name: str, severity: float = 1.0, seed: int = 42) -> Image.Image:
    """Apply a reproducible camera-like corruption without changing crop dimensions.

    Severity 1 is a moderate development setting, not a camera-calibrated blur
    level. The same function supports held-out deterministic severity sweeps.
    """
    if not math.isfinite(severity) or severity <= 0:
        # Invalid severity settings must not silently produce clean test data.
        raise ValueError("Degradation severity must be finite and positive")
    # Convert modes once so JPEG and interpolation behave consistently.
    image = image.convert("RGB")
    # Local randomness makes crop translations reproducible and isolated from training.
    generator = random.Random(seed)
    if name == "clean":
        return image.copy()
    if name == "defocus":
        # Isotropic blur approximates focus errors; motion is tested separately.
        return image.filter(ImageFilter.GaussianBlur(radius=1.2 * severity))
    if name == "motion":
        # A five-pixel line kernel can be extended by downscaling before convolution.
        scale = max(0.1, min(1.0, 1.0 / severity))
        reduced = image.resize((max(5, round(image.width * scale)), max(5, round(image.height * scale))))
        # Horizontal and vertical motion remain repeatable for a chosen sample seed.
        is_horizontal = generator.choice([True, False])
        weights = [
            1.0 if (index // 5 == 2 if is_horizontal else index % 5 == 2) else 0.0 for index in range(25)
        ]
        return reduced.filter(ImageFilter.Kernel((5, 5), weights, scale=5)).resize(
            image.size, Image.Resampling.BICUBIC
        )
    if name == "downsample":
        # Lost native resolution cannot be recovered merely by enlarging the crop.
        scale = 1.0 / (1.0 + 2.0 * severity)
        reduced = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        return reduced.resize(image.size, Image.Resampling.BICUBIC)
    if name == "jpeg":
        # Encode/decode introduces real codec artifacts instead of an additive proxy.
        quality = max(5, min(95, round(80 / (1 + severity))))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")
    if name == "crop":
        # Simulate imperfect detector boundaries while retaining the original size.
        shift_x = generator.choice([-1, 1]) * image.width * min(0.25, 0.035 * severity)
        shift_y = generator.choice([-1, 1]) * image.height * min(0.25, 0.035 * severity)
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            Image.Resampling.BILINEAR,
            fillcolor=(127, 127, 127),
        )
    raise ValueError(f"Unknown degradation {name!r}")


def augment_pair(
    image: Image.Image, mask: Image.Image | None, generator: random.Random
) -> tuple[Image.Image, Image.Image | None]:
    """Move clean mask geometry with its image, then corrupt only image evidence."""
    # Small rotations and translations represent detector crop/perspective residuals.
    angle = generator.uniform(-5, 5)
    shift_x = generator.uniform(-0.025, 0.025) * image.width
    shift_y = generator.uniform(-0.04, 0.04) * image.height
    # Keep masks binary at geometric boundaries; image interpolation remains smooth.
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(127, 127, 127))
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        Image.Resampling.BILINEAR,
        fillcolor=(127, 127, 127),
    )
    if mask is not None:
        # Identical transforms prevent segmentation labels drifting from input glyphs.
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        mask = mask.transform(
            mask.size,
            Image.Transform.AFFINE,
            (1, 0, shift_x, 0, 1, shift_y),
            Image.Resampling.NEAREST,
            fillcolor=0,
        )
    # Vary illumination independently of the clean character segmentation target.
    image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.65, 1.3))
    image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.7, 1.3))
    for name in ("defocus", "motion", "downsample", "jpeg"):
        if generator.random() < 0.35:
            # Moderate sampled severity avoids making most training labels unreadable.
            image = apply_degradation(image, name, generator.uniform(0.25, 1.2), generator.randrange(2**31))
    return image, mask


class PlateDataset:
    """Lazily load crops and optional masks with training-only paired augmentation."""

    def __init__(
        self, records: list[PlateRecord], max_length: int = 7, augment: bool = False, seed: int = 42
    ):
        """Own a private RNG; loaders should use num_workers=0 for reproducibility."""
        # Records are copied so external list mutation cannot change an epoch.
        self.records = list(records)
        # Output labels have one fixed head per configured character position.
        self.max_length = max_length
        # Validation and test callers leave augmentation disabled.
        self.augment = augment
        # Base seed and epoch derive per-record augmentation without dependence on load order.
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select reproducible new augmentation for the next training epoch."""
        # Epoch zero is also the deterministic default for standalone dataset checks.
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of independently loaded crop rows."""
        # Dataset length counts variants; split isolation is handled before construction.
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        """Return tensors plus provenance; raise on unreadable/misaligned source files."""
        # Lazy imports keep manifest preparation usable without the ML runtime.
        import torch
        from anpr.model import prepare_image, prepare_mask

        # Selected record retains source labels for evaluation and debugging.
        record = self.records[index]
        with Image.open(record.path) as source_image:
            # Copy pixel data before closing the file-backed image object.
            image = source_image.convert("RGB")
        # Absent masks receive a placeholder tensor and an explicit supervision flag.
        mask = None
        if record.mask_path is not None:
            with Image.open(record.mask_path) as source_mask:
                mask = source_mask.convert("L")
            if mask.size != image.size:
                raise ValueError(f"Mask/image dimensions differ for {record.path}")
        if self.augment:
            # Paired geometry matters only for training; evaluation stays deterministic.
            identity = f"{self.seed}:{self.epoch}:{record.group}:{record.plate}:{record.path.name}"
            # SHA256 avoids Python's process-randomized hash and remains stable across runs.
            record_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            image, mask = augment_pair(image, mask, random.Random(record_seed))
        return {
            "image": prepare_image(image),
            "targets": torch.tensor(encode_plate(record.plate, self.max_length), dtype=torch.long),
            "mask": prepare_mask(mask if mask is not None else Image.new("L", image.size, 0)),
            "has_mask": mask is not None,
            "plate": record.plate,
            "path": str(record.path),
        }


def _synthetic_formats(formats: list[str] | None) -> tuple[list[str], int]:
    """Validate explicit L/D/A masks and bound their combined identity space.

    Duplicate or fully subsumed masks do not inflate the capacity check. The
    remaining sum is an upper bound when masks partially overlap; sampling also
    has a finite attempt budget so that ambiguity cannot create an infinite loop.
    """
    # An omitted layout preserves the original UK-style baseline rather than changing old experiments.
    requested_formats = ["LLDDLLL"] if formats is None else formats
    if isinstance(requested_formats, (str, bytes)) or not requested_formats:
        raise ValueError("formats must be a nonempty list of L/D/A masks")
    # Preserve user ordering while avoiding accidental duplicate sampling weights.
    unique_formats = []
    for format_mask in requested_formats:
        # Every position describes a supported character class, with no implicit padding token.
        if (
            not isinstance(format_mask, str)
            or not 1 <= len(format_mask) <= 32
            or any(character not in "LDA" for character in format_mask)
        ):
            raise ValueError("Each format must contain 1 through 32 L/D/A positions")
        if format_mask not in unique_formats:
            unique_formats.append(format_mask)
    # Capacity counts omit masks whose entire language is already supplied by a broader mask.
    capacity_formats = [
        format_mask
        for format_mask in unique_formats
        if not any(
            other_mask != format_mask
            and len(other_mask) == len(format_mask)
            and all(other == "A" or other == current for current, other in zip(format_mask, other_mask))
            for other_mask in unique_formats
        )
    ]
    # Letter and digit spaces are disjoint; A includes their combined 36 symbols.
    alphabet_sizes = {"L": 26, "D": 10, "A": len(CHARACTERS)}
    capacity_upper_bound = sum(
        math.prod(alphabet_sizes[position] for position in format_mask) for format_mask in capacity_formats
    )
    return unique_formats, capacity_upper_bound


def _synthetic_fonts(font_paths: list[str | Path] | None) -> tuple[list, list[dict]]:
    """Load the exact requested font resources before creating output files."""
    # The legacy monospaced font and fallback retain their original initial point sizes.
    default_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    if font_paths is None and not default_path.is_file():
        # Pillow's fallback has no independent font path, so provenance must say so explicitly.
        return [ImageFont.load_default(size=40)], [
            {"path": None, "sha256": None, "source": "pillow_builtin_default", "initial_size": 40}
        ]
    # Explicit pools must identify usable files instead of silently falling back to another glyph design.
    requested_paths = [default_path] if font_paths is None else font_paths
    if isinstance(requested_paths, (str, bytes, Path)) or not requested_paths:
        raise ValueError("font_paths must be a nonempty list of font files")
    # Font objects and their byte-level descriptors remain index-aligned during sampling.
    fonts = []
    font_descriptors = []
    # Canonical input paths identify repeated font entries without comparing large font objects.
    seen_paths = set()
    for supplied_path in requested_paths:
        # Resolve each file once so duplicate spellings do not change the sampling distribution.
        font_path = Path(supplied_path).resolve()
        if font_path in seen_paths:
            continue
        if not font_path.is_file():
            raise ValueError(f"Missing font file: {font_path}")
        try:
            # Loading now rejects malformed files before any dataset artifacts are written.
            font = ImageFont.truetype(str(font_path), 43)
        except OSError as error:
            raise ValueError(f"Unusable font file: {font_path}") from error
        fonts.append(font)
        font_descriptors.append(
            {
                "path": str(font_path),
                "sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
                "source": "font_file",
                "initial_size": 43,
            }
        )
        seen_paths.add(font_path)
    return fonts, font_descriptors


def _fit_synthetic_font(font, plate: str, drawing: ImageDraw.ImageDraw):
    """Reduce an optional font/layout to fit the source canvas without clipping glyphs."""
    # Eight-pixel margins leave room for detector-like crop changes during later augmentation.
    available_width, available_height = 240, 64
    fitted_font = font
    while True:
        # Bounding boxes include real glyph bearings rather than just nominal font advance.
        bounds = drawing.textbbox((0, 0), plate, font=fitted_font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if text_width <= available_width and text_height <= available_height:
            return fitted_font, bounds
        # Lowering the point size preserves font outline rendering and exact image/mask alignment.
        current_size = fitted_font.size
        fit_scale = min(available_width / max(1, text_width), available_height / max(1, text_height))
        next_size = max(1, min(current_size - 1, math.floor(current_size * fit_scale)))
        if next_size == current_size:
            raise ValueError("Requested format cannot fit within the synthetic rendering canvas")
        fitted_font = fitted_font.font_variant(size=next_size)


def generate_synthetic_dataset(
    output_dir: str | Path,
    count: int = 300,
    seed: int = 42,
    *,
    formats: list[str] | None = None,
    font_paths: list[str | Path] | None = None,
) -> Path:
    """Render seeded plate identities and aligned masks with explicit format/font coverage.

    Count is the number of distinct identities; each produces clean and defocused
    crops. L means letters, D means digits and A means all 36 supported symbols.
    Shorter formats teach trailing padding when used with a larger-capacity model.
    An omitted format/font pool reproduces the original LLDDLLL baseline exactly.
    Filesystem writes begin only after configuration and label planning succeed.
    Synthetic rendering still does not establish legal typography or camera-domain
    accuracy; generator.json records that limitation and the exact font resources.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 3:
        # Three independent identities are the minimum for three isolated splits.
        raise ValueError("Generate at least three synthetic plate identities")
    # All generated files remain under the caller-selected output folder.
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        # Generation must not overwrite a real manifest or previously corrected OCR labels.
        raise FileExistsError(f"Synthetic output exists and is not empty: {output_dir}")
    # Validate finite label coverage and every font before a destination becomes visible.
    format_masks, capacity_upper_bound = _synthetic_formats(formats)
    fonts, font_descriptors = _synthetic_fonts(font_paths)
    if count > capacity_upper_bound:
        raise ValueError(f"Requested count {count} exceeds the format identity space {capacity_upper_bound}")
    # A private seeded renderer makes both labels and corruptions reproducible.
    generator = random.Random(seed)
    # Plans retain sampled font/color decisions in the same order as the legacy renderer's RNG calls.
    render_plans = []
    # Distinct registration text defines an identity even when several formats can generate it.
    used_plates: set[str] = set()
    # Partial overlaps can make the union smaller than its upper bound; cap retries before writing anything.
    attempts_remaining = max(1000, count * 1000)
    while len(used_plates) < count:
        if attempts_remaining == 0:
            raise ValueError("Requested identity count could not be sampled; reduce count or expand formats")
        attempts_remaining -= 1
        # Avoid drawing an extra random number for a singleton, preserving original seed replay.
        format_mask = generator.choice(format_masks) if len(format_masks) > 1 else format_masks[0]
        plate = "".join(
            generator.choice(
                CHARACTERS[:26] if position == "L" else CHARACTERS[26:] if position == "D" else CHARACTERS
            )
            for position in format_mask
        )
        if plate in used_plates:
            continue
        used_plates.add(plate)
        # Font index selects one identity's glyph design; background retains the two legacy pale colors.
        font_index = generator.randrange(len(fonts)) if len(fonts) > 1 else 0
        background = (245, 205, 45) if generator.random() < 0.5 else (240, 240, 235)
        render_plans.append((plate, font_index, background))
    # Plans are complete and usable before creating even the destination directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Records collect paired supervised rows, while font_usage counts identities per source font.
    records: list[PlateRecord] = []
    font_usage = {descriptor["path"] or descriptor["source"]: 0 for descriptor in font_descriptors}
    for plate, font_index, background in render_plans:
        # Drawing at the final source resolution keeps foreground targets exact.
        mask = Image.new("L", (256, 80), 0)
        drawing = ImageDraw.Draw(mask)
        # The default path intentionally retains its old geometry; configurable pools fit wider labels safely.
        font = fonts[font_index]
        if formats is not None or font_paths is not None:
            font, bounds = _fit_synthetic_font(font, plate, drawing)
        else:
            bounds = drawing.textbbox((0, 0), plate, font=font)
        # Usage identifies whether every requested typography option actually appeared in this finite set.
        font_descriptor = font_descriptors[font_index]
        font_usage[font_descriptor["path"] or font_descriptor["source"]] += 1
        # Text bounding boxes center visible glyphs rather than font ascender space.
        text_x = (mask.width - (bounds[2] - bounds[0])) // 2 - bounds[0]
        text_y = (mask.height - (bounds[3] - bounds[1])) // 2 - bounds[1]
        drawing.text((text_x, text_y), plate, fill=255, font=font)
        # Both standard background colors share identical glyph targets.
        image = Image.new("RGB", mask.size, background)
        image.paste((15, 15, 15), mask=mask)
        # Store one mask shared by clean/degraded variants of this identity.
        mask_path = output_dir / f"{plate}_mask.png"
        mask.save(mask_path)
        for mode in ("clean", "defocus"):
            # Label-derived filenames are acceptable only for this synthetic smoke set.
            image_path = output_dir / f"{plate}_{mode}.png"
            apply_degradation(image, mode, 0.7, seed).save(image_path)
            records.append(PlateRecord(image_path, plate, f"synthetic-{plate}", mask_path=mask_path))
    # Store explicit assignments to keep training/evaluation commands on identical splits.
    partitions = split_records(records, seed)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        # Relative paths make the generated folder movable as one artifact.
        writer = csv.DictWriter(manifest_file, fieldnames=["path", "plate", "group", "split", "mask_path"])
        writer.writeheader()
        for partition in partitions.values():
            for record in partition:
                writer.writerow(
                    {
                        "path": record.path.name,
                        "plate": record.plate,
                        "group": record.group,
                        "split": record.split,
                        "mask_path": record.mask_path.name,
                    }
                )
    # Provenance is separate from the legacy CSV so existing manifests and pixels stay byte-compatible.
    generation_metadata = {
        "rendering_version": "plate-renderer-v2",
        "pillow_version": pillow_version,
        "freetype_version": features.version_module("freetype2"),
        "seed": seed,
        "identity_count": count,
        "crop_count": len(records),
        "formats": format_masks,
        "fonts": font_descriptors,
        "font_usage": font_usage,
        "source_size": [256, 80],
        "variants": ["clean", "defocus"],
        "format_sampling": "uniform format, then uniform supported character per position; unique identities",
        "legacy_default_rendering": formats is None and font_paths is None,
        "limitations": [
            "Synthetic glyphs are not a substitute for real-camera training and held-out evaluation",
            "Explicit fonts do not establish legal registration typography or jurisdiction coverage",
            "Finite samples need a positional class-coverage audit before training",
        ],
    }
    (output_dir / "generator.json").write_text(json.dumps(generation_metadata, indent=2) + "\n")
    return manifest_path


def main() -> None:
    """Generate a labelled synthetic smoke dataset without external credentials."""
    # A small CLI makes synthetic setup reproducible in documentation and CI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300, help="Distinct identities, with two crops each")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--formats", nargs="+", help="Explicit L/D/A masks, for example A AA AAAAAAA")
    parser.add_argument("--font-paths", type=Path, nargs="+", help="Font files to sample per identity")
    # Parsed settings control only local synthetic file generation.
    arguments = parser.parse_args()
    print(
        generate_synthetic_dataset(
            arguments.generate_dir,
            arguments.count,
            arguments.seed,
            formats=arguments.formats,
            font_paths=arguments.font_paths,
        )
    )


if __name__ == "__main__":
    # Importing data utilities must never generate images or change files.
    main()
