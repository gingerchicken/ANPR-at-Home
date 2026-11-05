"""Check leakage controls and paired image preparation before training models."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, __version__ as pillow_version, features

from anpr.alphabet import CHARACTERS, encode_plate

from anpr.data import (
    PlateRecord,
    PlateDataset,
    apply_degradation,
    generate_synthetic_dataset,
    load_manifest,
    split_records,
)


def test_source_and_identity_connections_cannot_cross_splits(tmp_path):
    """A transitive source/identity chain must remain one indivisible component."""
    # The first three records form a chain through both source and plate identity.
    records = [
        PlateRecord(tmp_path / "a.png", "AB12CDE", "clip-a"),
        PlateRecord(tmp_path / "b.png", "CD34EFG", "clip-a"),
        PlateRecord(tmp_path / "c.png", "CD34EFG", "clip-b"),
        PlateRecord(tmp_path / "d.png", "EF56GHI", "clip-c"),
        PlateRecord(tmp_path / "e.png", "GH78IJK", "clip-d"),
    ]
    # Repeating a seeded assignment must preserve the exact split membership.
    partitions = split_records(records, seed=17)
    assert partitions == split_records(records, seed=17)
    assert all(partitions[split_name] for split_name in ("train", "val", "test"))
    assert any(
        {record.path for record in records[:3]}.issubset({record.path for record in partition})
        for partition in partitions.values()
    )


def test_explicit_splits_reject_transitive_leakage(tmp_path):
    """Explicit user splits cannot override source or identity isolation."""
    # Sharing a source across differently labelled records still constitutes leakage.
    records = [
        PlateRecord(tmp_path / "a.png", "AB12CDE", "clip-a", "train"),
        PlateRecord(tmp_path / "b.png", "CD34EFG", "clip-a", "test"),
    ]
    with pytest.raises(ValueError, match="leakage"):
        split_records(records)


def test_manifest_resolves_paths_and_rejects_unsupported_labels(tmp_path):
    """Manifest labels must be checked before a potentially long training run."""
    # The image path is relative to its manifest, independent of the process cwd.
    image_path = tmp_path / "plate.png"
    Image.new("RGB", (192, 64), "white").save(image_path)
    # The manifest initially contains a normalized, supported seven-character label.
    manifest_path = tmp_path / "plates.csv"
    manifest_path.write_text("path,plate,group\nplate.png,ab12 cde,clip-a\n")
    # Loading normalizes spacing while retaining the resolved local image path.
    records = load_manifest(manifest_path)
    assert records[0].path == image_path
    assert records[0].plate == "AB12CDE"
    manifest_path.write_text("path,plate,group\nplate.png,AB12CDEF,clip-a\n")
    with pytest.raises(ValueError, match="length|long|capacity"):
        load_manifest(manifest_path)


def test_partial_explicit_splits_are_rejected(tmp_path):
    """Mixing automatic and manual splits hides evaluation contamination."""
    # One unspecified row must not be silently assigned alongside explicit labels.
    records = [
        PlateRecord(tmp_path / "a.png", "AB12CDE", "a", "train"),
        PlateRecord(tmp_path / "b.png", "CD34EFG", "b"),
    ]
    with pytest.raises(ValueError, match="all|every"):
        split_records(records)


def test_manifest_rejects_conflicting_labels_for_one_image(tmp_path):
    """A duplicated path cannot become two different apparent plate identities."""
    # One file with inconsistent labels otherwise risks both bad supervision and leakage.
    Image.new("RGB", (192, 64), "white").save(tmp_path / "plate.png")
    # Different capture groups cannot make the underlying duplicated image independent.
    manifest_path = tmp_path / "conflict.csv"
    manifest_path.write_text("path,plate,group\nplate.png,AB12CDE,a\nplate.png,CD34EFG,b\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_manifest(manifest_path)


@pytest.mark.parametrize("name", ["clean", "defocus", "motion", "downsample", "jpeg", "crop"])
def test_degradation_preserves_size_and_is_reproducible(name):
    """Severity sweeps must apply identical corruptions when rerun."""
    # A non-uniform crop makes stochastic image transformations observable.
    image = Image.new("RGB", (192, 64), "white")
    image.paste("black", (40, 20, 80, 50))
    # Identical inputs and seeds must yield byte-identical degraded crops.
    first = apply_degradation(image, name, severity=1.0, seed=123)
    second = apply_degradation(image, name, severity=1.0, seed=123)
    assert first.size == image.size
    assert first.tobytes() == second.tobytes()


def test_generated_masks_and_splits_match_plate_crops(tmp_path):
    """Synthetic smoke data provides aligned glyph masks and isolated identities."""
    # Each generated identity has clean and blurred variants sharing one glyph mask.
    manifest_path = generate_synthetic_dataset(tmp_path, count=9, seed=4)
    records = load_manifest(manifest_path)
    assert len(records) == 18
    assert all(split_records(records).values())
    # Every stored mask must retain the image dimensions and contain foreground strokes.
    for record in records:
        # PIL contexts close files promptly even when assertions fail.
        with Image.open(record.path) as image, Image.open(record.mask_path) as mask:
            assert image.size == mask.size
            assert mask.getextrema() == (0, 255)
    # Reading raw rows also proves that the generator records its synthetic provenance.
    with manifest_path.open(newline="") as manifest_file:
        assert all(row["group"].startswith("synthetic-") for row in csv.DictReader(manifest_file))


def test_dataset_prepares_aligned_masks_and_seeded_training_crops(tmp_path):
    """Loading order must not alter reproducible per-record training augmentation."""
    # Small synthetic fixtures contain masks and image paths without external downloads.
    records = load_manifest(generate_synthetic_dataset(tmp_path, count=3, seed=5))
    dataset = PlateDataset(records, augment=True, seed=8)
    # Repeated reads of the same epoch must use the same crop and clean glyph geometry.
    first = dataset[0]
    dataset[1]
    repeated = dataset[0]
    assert first["image"].shape == (3, 64, 192)
    assert first["mask"].shape == (1, 64, 192)
    assert first["has_mask"]
    assert first["targets"].shape == (7,)
    assert first["image"].equal(repeated["image"])
    assert first["mask"].equal(repeated["mask"])
    # A new epoch deliberately chooses a different deterministic corruption.
    dataset.set_epoch(1)
    assert not first["image"].equal(dataset[0]["image"])


def test_synthetic_generation_does_not_overwrite_existing_labels(tmp_path):
    """A mistyped output directory must not replace a hand-labelled manifest."""
    # This existing manifest stands in for user annotations that cannot be regenerated.
    existing_manifest = tmp_path / "manifest.csv"
    existing_manifest.write_text("trusted labels\n")
    with pytest.raises(FileExistsError, match="not empty"):
        generate_synthetic_dataset(tmp_path, count=3)
    assert existing_manifest.read_text() == "trusted labels\n"


def test_default_generator_replays_the_legacy_manifest_and_pixels(tmp_path):
    """Existing seeds must retain their labels, assignments and baseline rendering."""
    # This manifest fingerprint was frozen before configurable layouts or fonts were introduced.
    manifest_path = generate_synthetic_dataset(tmp_path, count=9, seed=4)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "c8c0b997789d292dede624078c915b5656df03c2765e24ac51fe7e0cb25f9660"
    )
    # Pixel snapshots apply only to the exact font and rasterizer that produced the reference run.
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    if (
        not font_path.is_file()
        or pillow_version != "13.0.0"
        or features.version_module("freetype2") != "2.14.3"
        or hashlib.sha256(font_path.read_bytes()).hexdigest()
        != "3a3c502eeff669a231549e80df9f7c49de109bafe303170409e905d0b31a38fe"
    ):
        return
    # Selected images expose changed random consumption, background colors and text placement.
    expected_pixels = {
        "BO48RUP_clean.png": "7d3c5e90be170b886bb14927912ebb4f813b72905772295e800740cd9292ead5",
        "HJ16PEC_clean.png": "406da52c69c49e78ba487d2c9d6c8a2de4670063025b7a19a196caddfbe1517a",
    }
    for image_name, expected_digest in expected_pixels.items():
        with Image.open(manifest_path.parent / image_name) as image:
            assert hashlib.sha256(image.tobytes()).hexdigest() == expected_digest


def test_generic_format_can_render_every_alphanumeric_class_and_trailing_padding(tmp_path):
    """A generic character mask must not retain the baseline's letter/digit position bias."""
    # Exhausting a one-character format verifies all 36 supported symbols rather than a lucky sample.
    records = load_manifest(generate_synthetic_dataset(tmp_path, count=36, seed=8, formats=["A"]))
    assert {record.plate for record in records} == set(CHARACTERS)
    # Short labels keep one real class followed by six absent-character targets.
    assert all(encode_plate(record.plate)[1:] == [0] * 6 for record in records)


def test_variable_formats_and_fonts_replay_labels_pixels_and_metadata(tmp_path):
    """Seeded multi-font generation retains aligned clean targets across output folders."""
    # Both font files are intentionally explicit, so their bytes can identify the rendering inputs.
    font_paths = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    if not all(font_path.is_file() for font_path in font_paths):
        pytest.skip("The two explicit rendering fixture fonts are unavailable")
    # Different lengths exercise padding and long text fitting without changing model capacity silently.
    first_manifest = generate_synthetic_dataset(
        tmp_path / "first", count=30, seed=18, formats=["A", "AAA", "AAAAAAA"], font_paths=font_paths
    )
    second_manifest = generate_synthetic_dataset(
        tmp_path / "second", count=30, seed=18, formats=["A", "AAA", "AAAAAAA"], font_paths=font_paths
    )
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    # Generator metadata states the exact font bytes, requested label spaces and source resolution.
    metadata = json.loads((first_manifest.parent / "generator.json").read_text())
    assert metadata["formats"] == ["A", "AAA", "AAAAAAA"]
    assert metadata["source_size"] == [256, 80]
    assert {font["sha256"] for font in metadata["fonts"]} == {
        hashlib.sha256(font_path.read_bytes()).hexdigest() for font_path in font_paths
    }
    assert all(count > 0 for count in metadata["font_usage"].values())
    # Every record retains the original-size clean foreground map used to composite its glyphs.
    records = load_manifest(first_manifest)
    assert {len(record.plate) for record in records} == {1, 3, 7}
    for record in records:
        with (
            Image.open(record.path) as first_image,
            Image.open(second_manifest.parent / record.path.name) as second_image,
        ):
            assert first_image.tobytes() == second_image.tobytes()
        if record.path.stem.endswith("_clean"):
            with Image.open(record.path) as image, Image.open(record.mask_path) as mask:
                assert image.size == mask.size == (256, 80)
                assert mask.getbbox() is not None
                # Foreground pixels are darker than their known pale background at the same coordinates.
                foreground_x, foreground_y = next(
                    (x, y)
                    for y in range(mask.height)
                    for x in range(mask.width)
                    if mask.getpixel((x, y)) == 255
                )
                assert max(image.getpixel((foreground_x, foreground_y))) == 15


@pytest.mark.parametrize("formats", [[], [""], ["X"], ["a"], ["A" * 33]])
def test_invalid_synthetic_formats_fail_before_output_writes(tmp_path, formats):
    """Invalid layouts must fail before a half-generated dataset becomes visible."""
    # The child path does not yet exist and must stay absent on configuration errors.
    output_path = tmp_path / "invalid"
    with pytest.raises(ValueError, match="format"):
        generate_synthetic_dataset(output_path, count=3, formats=formats)
    assert not output_path.exists()


def test_impossible_label_count_and_invalid_fonts_fail_before_output_writes(tmp_path):
    """Finite label spaces and unusable font pools cannot start an unbounded render loop."""
    # A digit-only single position has exactly ten possible identities.
    output_path = tmp_path / "invalid"
    with pytest.raises(ValueError, match="identit|count|space"):
        generate_synthetic_dataset(output_path, count=11, formats=["D"])
    # A subset format must not inflate the capacity of a broader format that already contains it.
    with pytest.raises(ValueError, match="identit|count|space"):
        generate_synthetic_dataset(output_path, count=37, formats=["L", "A"])
    # A real file containing non-font bytes must fail loading rather than becoming a fallback font.
    malformed_font_path = tmp_path / "malformed.ttf"
    malformed_font_path.write_bytes(b"not a font")
    # Explicit font pools require at least one usable, real font file.
    for font_paths in ([], [tmp_path / "missing.ttf"], [malformed_font_path]):
        with pytest.raises(ValueError, match="font"):
            generate_synthetic_dataset(output_path, count=3, font_paths=font_paths)
    assert not output_path.exists()


def test_long_requested_layout_fits_the_rendering_canvas(tmp_path):
    """Supporting larger model capacities must not silently clip the rendered label."""
    # A 32-character layout cannot fit at the legacy 43-pixel font size without adjustment.
    manifest_path = generate_synthetic_dataset(tmp_path, count=3, seed=7, formats=["L" * 32])
    for record in load_manifest(manifest_path, max_length=32):
        with Image.open(record.mask_path) as mask:
            # All glyph ink remains inside the explicit eight-pixel canvas margin.
            left, top, right, bottom = mask.getbbox()
            assert left >= 8 and top >= 8 and right <= 248 and bottom <= 72
