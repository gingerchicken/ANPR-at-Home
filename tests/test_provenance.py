"""Prevent changing an evaluation manifest/seed from manufacturing held-out results."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from anpr.evaluate import validate_checkpoint_splits, validate_calibration_test_overlap


@pytest.mark.parametrize(
    "plate,group,path",
    [
        ("CC33CCC", "another-session", "/different.png"),
        ("DD44DDD", "new-calibration-session", "/different.png"),
        ("DD44DDD", "another-session", "/calibration.png"),
    ],
)
def test_transferred_calibration_records_cannot_become_test_rows(plate, group, path):
    """New calibration identities need protection even when absent from checkpoint development data."""
    # The checkpoint never saw CC33CCC; it was fitted later in a separate calibration source.
    metadata = {
        "split_provenance": {
            "train": [{"plate": "AA11AAA", "group": "train", "path": "/train.png"}],
            "val": [{"plate": "BB22BBB", "group": "val", "path": "/val.png"}],
        }
    }
    calibration = {
        "validation_provenance": [
            {"plate": "CC33CCC", "group": "new-calibration-session", "path": "/calibration.png"},
        ]
    }
    test_records = [SimpleNamespace(plate=plate, group=group, path=Path(path))]
    # Original-checkpoint isolation alone passes, reproducing the missing transfer check.
    validate_checkpoint_splits(metadata, {"test": test_records})
    with pytest.raises(ValueError, match="[Cc]alibration.*leakage|overlap"):
        validate_calibration_test_overlap(calibration, test_records)


def test_evaluation_rejects_checkpoint_training_identity_in_new_test_manifest():
    """A renamed crop from another manifest still exposes the same training identity."""
    # Absolute path changes do not erase the fact that this plate trained the checkpoint.
    metadata = {
        "split_provenance": {
            "train": [{"plate": "AB12CDE", "group": "video1", "path": "/original/train.jpg"}],
            "val": [{"plate": "XY34FGH", "group": "video2", "path": "/original/val.jpg"}],
        }
    }
    splits = {"test": [SimpleNamespace(plate="AB12CDE", group="new-video", path=Path("/new.jpg"))]}
    with pytest.raises(ValueError, match="leakage"):
        validate_checkpoint_splits(metadata, splits)


def test_evaluation_rejects_source_used_for_checkpoint_selection():
    """A different plate from the same validation capture cannot become a test row."""
    # Shared camera/session conditions can leak even when plate strings differ.
    metadata = {
        "split_provenance": {
            "train": [{"plate": "AB12CDE", "group": "video1", "path": "/train.jpg"}],
            "val": [{"plate": "XY34FGH", "group": "video2", "path": "/val.jpg"}],
        }
    }
    splits = {"test": [SimpleNamespace(plate="ZZ99AAA", group="video2", path=Path("/test.jpg"))]}
    with pytest.raises(ValueError, match="leakage"):
        validate_checkpoint_splits(metadata, splits)


def test_evaluation_requires_provenance_and_accepts_independent_captures():
    """Weights alone cannot prove held-out membership; independent new rows can qualify."""
    # This test row differs in plate, source and file path from both development splits.
    splits = {"test": [SimpleNamespace(plate="ZZ99AAA", group="video3", path=Path("/test.jpg"))]}
    with pytest.raises(ValueError, match="provenance"):
        validate_checkpoint_splits({}, splits)
    validate_checkpoint_splits(
        {
            "split_provenance": {
                "train": [{"plate": "AB12CDE", "group": "video1", "path": "/train.jpg"}],
                "val": [{"plate": "XY34FGH", "group": "video2", "path": "/val.jpg"}],
            }
        },
        splits,
    )
