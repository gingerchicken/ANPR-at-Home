"""Check honest labels, source isolation and original-pixel benchmark crops."""

import csv
import json

from PIL import Image
import pytest

from anpr.openalpr import prepare_openalpr


def write_source(tmp_path, annotation):
    """Create a tiny local benchmark fixture without downloading external data."""
    # The source color makes accidental overlays or channel swapping visible.
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (100, 50), (20, 40, 200)).save(source / "scene.png")
    (source / "scene.txt").write_text(annotation)
    return source


def test_crops_original_pixels_and_normalizes_display_separators(tmp_path):
    """Image coordinates and label text must survive independent crop preparation."""
    # A spaced label denotes one seven-character plate, not eight output heads.
    source = write_source(tmp_path, "scene.png\t10\t5\t30\t20\tAB 12 CDE\n")
    (source / "LICENSE").write_text("upstream license text\n")
    manifest_path = prepare_openalpr(source, tmp_path / "crops")
    with manifest_path.open(newline="") as manifest_file:
        # Exactly one source box yields one labelled, held-out crop.
        rows = list(csv.DictReader(manifest_file))
    assert len(rows) == 1
    assert rows[0]["plate"] == "AB12CDE"
    assert rows[0]["group"] == "openalpr-eu"
    assert rows[0]["split"] == "test"
    with Image.open(manifest_path.parent / rows[0]["path"]) as crop:
        assert crop.size == (30, 20)
        assert crop.getpixel((0, 0)) == (20, 40, 200)
    assert (manifest_path.parent / "LICENSE").read_text() == "upstream license text\n"


def test_multiple_plate_rows_and_clipped_boxes_remain_distinct(tmp_path):
    """One frame may contain several plates, all kept together as held-out data."""
    # The first rectangle extends left of the image and must be clipped to valid pixels.
    source = write_source(tmp_path, "scene.png\t-5\t0\t30\t20\tAA11AAA\nscene.png\t50\t10\t40\t30\tBB22BBB\n")
    manifest_path = prepare_openalpr(source, tmp_path / "crops")
    with manifest_path.open(newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    assert len(rows) == 2
    assert rows[0]["path"] != rows[1]["path"]
    with Image.open(manifest_path.parent / rows[0]["path"]) as crop:
        assert crop.size == (25, 20)


def test_exclusions_retain_original_labels_and_reasons(tmp_path):
    """Unsupported labels and capacity limits must never be repaired by truncation."""
    # A valid row keeps the manifest usable while two excluded rows remain auditable.
    source = write_source(
        tmp_path,
        "scene.png\t0\t0\t30\t20\tAA11AAA\nscene.png\t0\t0\t30\t20\tAB123456\nscene.png\t0\t0\t30\t20\tAB*1234\n",
    )
    manifest_path = prepare_openalpr(source, tmp_path / "crops")
    provenance = json.loads((manifest_path.parent / "provenance.json").read_text())
    assert provenance["included_count"] == 1
    assert provenance["excluded_count"] == 2
    assert [row["raw_plate"] for row in provenance["exclusions"]] == ["AB123456", "AB*1234"]
    assert [row["reason"] for row in provenance["exclusions"]] == [
        "exceeds_model_capacity",
        "unsupported_label",
    ]


@pytest.mark.parametrize("coordinates", ["nan\t0\t30\t20", "0\t0\t0\t20", "200\t0\t30\t20"])
def test_invalid_geometry_fails_before_writing_outputs(tmp_path, coordinates):
    """Malformed benchmark boxes must not produce partial or empty pixel crops."""
    # The destination is absent before and after rejecting unusable source geometry.
    source = write_source(tmp_path, f"scene.png\t{coordinates}\tAA11AAA\n")
    output = tmp_path / "crops"
    with pytest.raises(ValueError, match="coordinate|area|width|finite"):
        prepare_openalpr(source, output)
    assert not output.exists()


def test_source_escape_and_missing_images_are_rejected(tmp_path):
    """Annotation paths must stay inside the supplied benchmark folder."""
    # A real outside image must not become an implicit importer input.
    source = write_source(tmp_path, "../outside.png\t0\t0\t20\t20\tAA11AAA\n")
    Image.new("RGB", (100, 50)).save(tmp_path / "outside.png")
    with pytest.raises(ValueError, match="relative|outside|escape"):
        prepare_openalpr(source, tmp_path / "crops")
    (source / "scene.txt").write_text("missing.png\t0\t0\t20\t20\tAA11AAA\n")
    with pytest.raises(ValueError, match="missing|Missing"):
        prepare_openalpr(source, tmp_path / "crops")


def test_existing_outputs_and_symlink_sources_are_protected(tmp_path):
    """Existing human work and source-root boundaries survive importer failures."""
    # A label file outside the source root must not be reached through a symlink.
    source = write_source(tmp_path, "scene.png\t0\t0\t20\t20\tAA11AAA\n")
    output = tmp_path / "crops"
    output.mkdir()
    (output / "manifest.csv").write_text("trusted labels\n")
    with pytest.raises(FileExistsError):
        prepare_openalpr(source, output)
    assert (output / "manifest.csv").read_text() == "trusted labels\n"
    outside_annotation = tmp_path / "outside.txt"
    outside_annotation.write_text((source / "scene.txt").read_text())
    (source / "scene.txt").unlink()
    (source / "scene.txt").symlink_to(outside_annotation)
    with pytest.raises(ValueError, match="outside|escape"):
        prepare_openalpr(source, tmp_path / "another-output")
