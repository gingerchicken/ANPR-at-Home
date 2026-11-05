"""Verify crop routing and color order without relying on downloaded YOLO weights."""

from types import SimpleNamespace
import hashlib
import json
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from anpr.inference import PlateRecognizer, recognize_yolo_result


class FixedRecognizer(torch.nn.Module):
    """Supply known character evidence so integration tests check decoding behavior."""

    # Two positional heads let the fixture expose repeated-character and rank behavior.
    max_length = 2

    def forward(self, images):
        """Make AA the best plate and BA the next one, retaining repeated letters."""
        # Each batch example receives the same deterministic two-position evidence.
        logits = torch.full((images.shape[0], 2, 37), -10.0, device=images.device)
        logits[:, :, 1] = 5
        logits[:, 0, 2] = 4
        return logits


class FixedCTCRecognizer(torch.nn.Module):
    """Expose an A-blank-A path so repeated letters require actual CTC decoding."""

    # Capacity counts output characters, while the fixture supplies three alignment frames.
    max_length = 2
    model_kind = "ctc"

    def forward(self, images):
        """Supply strongly separated A emissions without positional-padding semantics."""
        # A blank between equal characters distinguishes AA from a single repeated-frame A.
        logits = torch.full((images.shape[0], 3, 37), -12.0, device=images.device)
        logits[:, 0, 1] = 12
        logits[:, 1, 0] = 12
        logits[:, 2, 1] = 12
        return logits


def test_ctc_recognizer_uses_alignment_scores_and_retains_full_frame_distributions():
    """Generic crop inference must dispatch by model kind instead of treating frames as slots."""
    # The same crop API handles both model families without changing the original positional decoder.
    recognizer = PlateRecognizer(FixedCTCRecognizer())
    prediction = recognizer.recognize(Image.new("RGB", (200, 40), "white"), top_k=2)
    assert prediction["candidates"][0]["plate"] == "AA"
    assert len(prediction["positions"]) == 2
    assert len(prediction["frames"]) == 3
    assert all(len(frame["alternatives"]) == 37 for frame in prediction["frames"])
    assert prediction["metadata"]["model_kind"] == "ctc"
    assert (
        prediction["metadata"]["position_probability_kind"] == "ctc_ordered_output_marginal_including_absence"
    )
    assert len(prediction["alignment_positions"]) == 2
    with pytest.raises(ValueError, match="capacity"):
        # Extra frames are not permission to decode registrations longer than the trained capacity.
        recognizer.recognize(Image.new("RGB", (200, 40), "white"), max_length=3)


def test_standalone_crop_keeps_alternatives_and_uncertainty():
    """Inference must preserve character alternatives rather than return argmax alone."""
    # The stub isolates image routing and decoding from untrained network accuracy.
    recognizer = PlateRecognizer(FixedRecognizer())
    prediction = recognizer.recognize(Image.new("RGB", (200, 40), "white"), top_k=2)
    assert [candidate["plate"] for candidate in prediction["candidates"]] == ["AA", "BA"]
    assert len(prediction["positions"][0]["alternatives"]) == 37
    assert prediction["status"] == "uncertain"
    assert prediction["latency_ms"] >= 0


def test_inference_keeps_training_provenance_out_of_public_predictions():
    """Video output must not repeat or expose unrelated training labels and paths."""
    # Internal provenance is useful for leakage audits but irrelevant to a crop consumer.
    recognizer = PlateRecognizer(
        FixedRecognizer(),
        metadata={
            "max_length": 2,
            "preprocess": "test",
            "manifest": "/private/labels.csv",
            "split_provenance": {"train": [{"plate": "PRIVATE", "path": "/private/crop.jpg"}]},
        },
    )
    prediction = recognizer.recognize(Image.new("RGB", (200, 40), "white"), top_k=1)
    assert prediction["model_metadata"] == {"max_length": 2, "preprocess": "test"}
    assert "split_provenance" in recognizer.metadata


def test_standalone_recognition_applies_checkpoint_bound_validation_calibration(tmp_path):
    """Crop probabilities must use the selected model's frozen validation scale."""
    # A tiny file identifies the checkpoint; a stub forward isolates inference from model loading.
    checkpoint_path = tmp_path / "weights.pt"
    checkpoint_path.write_bytes(b"specific model weights")
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "temperature": 2.0,
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "manifest_sha256": "a" * 64,
                "fit_split": "val",
                "validation_sample_count": 1,
                "validation_provenance": [{"plate": "AB", "group": "validation", "path": "/val.png"}],
                "whole_plate_confidence_calibrated": False,
            }
        )
    )
    with mock.patch("anpr.inference.load_checkpoint", return_value=(FixedRecognizer(), {"temperature": 1.0})):
        # The shared validator binds the scale to exact weights before any crop is recognized.
        recognizer = PlateRecognizer.from_checkpoint(checkpoint_path, calibration=calibration_path)
    prediction = recognizer.recognize(Image.new("RGB", (200, 40), "white"), top_k=2)
    assert prediction["metadata"]["temperature"] == 2.0
    assert prediction["temperature_source"] == "validation_sidecar"
    # A deliberate per-call override must not be represented as the validation-fitted scale.
    override_prediction = recognizer.recognize(Image.new("RGB", (200, 40), "white"), temperature=1.0)
    assert override_prediction["temperature_source"] == "explicit_override"


def test_yolo_crops_original_pixels_converts_bgr_and_keeps_box_metadata():
    """Annotated result plots must never become recognition input."""
    # Red in BGR checks the channel conversion; only one box is the plate class.
    bgr_image = np.zeros((20, 40, 3), dtype=np.uint8)
    bgr_image[:, :, 2] = 255
    detection = SimpleNamespace(
        orig_img=bgr_image,
        path="frame.jpg",
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[-2, 2, 12, 10], [0, 0, 20, 20]]),
            cls=torch.tensor([0, 1]),
            conf=torch.tensor([0.8, 0.99]),
        ),
    )
    # Capture the exact crop reaching OCR, without coupling this test to tensor normalization.
    captured_crops = []

    class CropRecorder:
        """Inspect the adapter's public crop interface."""

        def recognize(self, image, **options):
            """Record an independent crop so the caller cannot mutate it later."""
            captured_crops.append(image.copy())
            return {"status": "uncertain", "candidates": []}

    predictions = recognize_yolo_result(detection, CropRecorder(), padding_fraction=0)
    assert len(predictions) == 1
    assert captured_crops[0].size == (12, 8)
    assert captured_crops[0].getpixel((0, 0)) == (255, 0, 0)
    assert predictions[0]["crop_xyxy"] == [0, 2, 12, 10]
    assert predictions[0]["detection_confidence"] == pytest.approx(0.8)
    assert predictions[0]["source"] == "frame.jpg"


def test_yolo_rejects_nonfinite_boxes():
    """Malformed detector coordinates must fail before image slicing."""
    # Nonfinite coordinates cannot be represented as valid crop geometry.
    detection = SimpleNamespace(
        orig_img=np.zeros((20, 40, 3), dtype=np.uint8),
        path="frame.jpg",
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[float("nan"), 0, 12, 10]]), cls=torch.tensor([0]), conf=torch.tensor([0.8])
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        recognize_yolo_result(detection, None)


def test_yolo_rejects_nonfinite_confidence_before_export():
    """Python API results must remain valid JSON even when detector output is malformed."""
    # A confidence NaN cannot be interpreted as evidence or serialized as strict JSON.
    detection = SimpleNamespace(
        orig_img=np.zeros((20, 40, 3), dtype=np.uint8),
        path="frame.jpg",
        boxes=SimpleNamespace(
            xyxy=torch.tensor([[0, 0, 12, 10]]), cls=torch.tensor([0]), conf=torch.tensor([float("nan")])
        ),
    )
    with pytest.raises(ValueError, match="confidence"):
        recognize_yolo_result(detection, None)
