"""Protect source assets from mistaken report/checkpoint destinations."""

from argparse import Namespace
from unittest import mock

import pytest

from anpr.paths import protect_output_paths
from anpr.evaluate import main as evaluate_main
from anpr.train import run_training
from anpr.inference import main as inference_main


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_output_aliases_cannot_replace_input_assets(tmp_path, alias_kind):
    """Resolved names and inode aliases must all protect the original file bytes."""
    # The protected asset represents weights or labels that cannot be recovered from a report.
    source = tmp_path / "weights.pt"
    source.write_bytes(b"trusted weights")
    destination = source
    if alias_kind == "symlink":
        # A different visible filename may resolve to the protected source.
        destination = tmp_path / "report.json"
        destination.symlink_to(source)
    elif alias_kind == "hardlink":
        # Hard links share an inode even though their resolved paths differ.
        destination = tmp_path / "report.json"
        destination.hardlink_to(source)
    with pytest.raises(ValueError, match="alias|input"):
        protect_output_paths([source], [destination])
    assert source.read_bytes() == b"trusted weights"


def test_distinct_report_reruns_are_allowed_but_outputs_cannot_alias_each_other(tmp_path):
    """Existing separate reports remain replaceable without confusing two output roles."""
    # Neither pre-existing output has input ownership, so a normal rerun is allowed.
    source = tmp_path / "weights.pt"
    source.write_bytes(b"weights")
    report = tmp_path / "report.json"
    report.write_text("old report")
    protect_output_paths([source], [report, tmp_path / "calibration.json"])
    with pytest.raises(ValueError, match="alias"):
        protect_output_paths([source], [report, report.parent / "." / report.name])


def test_evaluation_rejects_checkpoint_destination_before_loading_or_writing(tmp_path):
    """The real CLI must reject destructive configuration before deserializing weights."""
    # These bytes intentionally are not a torch checkpoint: only the early path guard can pass this test.
    checkpoint = tmp_path / "only-weights.pt"
    checkpoint.write_bytes(b"trusted only weights")
    with pytest.raises(ValueError, match="alias|input"):
        evaluate_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(tmp_path / "labels.csv"),
                "--output",
                str(checkpoint),
            ]
        )
    assert checkpoint.read_bytes() == b"trusted only weights"


def test_training_protects_manifest_before_model_initialization(tmp_path):
    """A training checkpoint must not overwrite the manifest that labels its images."""
    # The invalid CSV content distinguishes early ownership validation from manifest parsing errors.
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("trusted labels\n")
    arguments = Namespace(
        epochs=1, batch_size=1, threads=1, learning_rate=0.0001, manifest=manifest, output=manifest, seed=42
    )
    with pytest.raises(ValueError, match="alias|input"):
        run_training(arguments)
    assert manifest.read_text() == "trusted labels\n"


@pytest.mark.parametrize("asset_kind", ["image", "mask"])
def test_training_does_not_replace_a_manifest_source_asset(tmp_path, asset_kind):
    """Dataset-owned images and masks need protection before the first checkpoint save."""
    # File bytes need not decode: the path guard must fail before the training dataset opens them.
    image = tmp_path / "crop.png"
    image.write_bytes(b"trusted crop")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"trusted mask")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,plate,group,split,mask_path\ncrop.png,AA11AAA,capture,train,mask.png\n")
    destination = image if asset_kind == "image" else mask
    arguments = Namespace(
        epochs=1,
        batch_size=1,
        threads=1,
        learning_rate=0.0001,
        manifest=manifest,
        output=destination,
        seed=42,
        max_length=7,
    )
    with pytest.raises(ValueError, match="alias|input"):
        run_training(arguments)
    assert image.read_bytes() == b"trusted crop"
    assert mask.read_bytes() == b"trusted mask"


@pytest.mark.parametrize("source_mode", ["--image", "--source"])
def test_inference_does_not_replace_its_image_input(tmp_path, source_mode):
    """Standalone crops and detector source files remain intact after invalid output selection."""
    # The file is deliberately undecodable so ownership must be checked before loading any model.
    image = tmp_path / "frame.png"
    image.write_bytes(b"trusted image")
    arguments = [
        "anpr.inference",
        "--checkpoint",
        str(tmp_path / "weights.pt"),
        source_mode,
        str(image),
        "--output",
        str(image),
    ]
    if source_mode == "--source":
        # Detector mode needs weights in its declared source configuration, but must not load them here.
        arguments.extend(["--yolo-weights", str(tmp_path / "detector.pt")])
    with mock.patch("sys.argv", arguments), pytest.raises(ValueError, match="alias|input"):
        inference_main()
    assert image.read_bytes() == b"trusted image"


@pytest.mark.parametrize("source_pattern", ["directory", "glob", "literal_metacharacters"])
def test_detector_directory_and_glob_inputs_protect_their_source_images(tmp_path, source_pattern):
    """A result destination inside a source collection must not overwrite one of its frames."""
    # This is a media input even though the CLI has not yet decoded its deliberately invalid bytes.
    image = tmp_path / ("frame[1].png" if source_pattern == "literal_metacharacters" else "frame.png")
    image.write_bytes(b"trusted frame")
    source = str(tmp_path) if source_pattern == "directory" else str(tmp_path / "*.png")
    if source_pattern == "literal_metacharacters":
        # A glob match must not hide the literal file that the detector was asked to open.
        (tmp_path / "frame1.png").write_bytes(b"different frame")
        source = str(image)
    arguments = [
        "anpr.inference",
        "--checkpoint",
        str(tmp_path / "weights.pt"),
        "--yolo-weights",
        str(tmp_path / "detector.pt"),
        "--source",
        source,
        "--output",
        str(image),
    ]
    with mock.patch("sys.argv", arguments), pytest.raises(ValueError, match="alias|input"):
        inference_main()
    assert image.read_bytes() == b"trusted frame"
