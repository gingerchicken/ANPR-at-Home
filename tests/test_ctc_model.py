"""Check sequence alignment, blank semantics and CTC checkpoint behavior."""

from argparse import Namespace
import math

import pytest
import torch

from anpr.alphabet import encode_plate
from anpr.ctc_model import PlateCTC
from anpr.ctc_train import ctc_batch_loss, minimum_ctc_frames, prepare_ctc_targets, run_training
from anpr.model import load_checkpoint, save_checkpoint


def test_sequence_model_retains_time_axis_and_all_character_classes():
    """A seven-character capacity must not reduce the encoder to seven CTC frames."""
    # Repeated labels need extra blank frames, so the real encoder must keep its 24-cell width.
    torch.set_num_threads(1)
    model = PlateCTC(max_length=7).eval()
    with torch.inference_mode():
        # The normalized input contract is shared with the positional baseline.
        logits = model(torch.zeros(2, 3, 64, 192))
    assert logits.shape == (2, 24, 37)
    assert model.model_kind == "ctc"
    assert model.time_steps == 24
    assert model.use_mask is False
    assert model.head_layout == "ctc"


def test_ctc_targets_preserve_repeated_characters_and_remove_only_trailing_padding():
    """AA is two target symbols, while zero belongs only to the CTC frame vocabulary."""
    # Training uses flat, unpadded labels rather than collapsing repeated target characters.
    plates = ["AA", "B12"]
    padded_targets = torch.tensor([encode_plate(plate, 4) for plate in plates])
    flat_targets, target_lengths = prepare_ctc_targets(padded_targets, plates, time_steps=5)
    assert flat_targets.tolist() == [1, 1, 2, 28, 29]
    assert target_lengths.tolist() == [2, 3]
    assert not torch.any(flat_targets == 0)
    assert minimum_ctc_frames("AA") == 3
    assert minimum_ctc_frames("AAA") == 5
    assert minimum_ctc_frames("ABBA") == 5


def test_repeated_character_alignment_is_finite_when_blank_separation_is_possible():
    """The loss must sum valid CTC alignments instead of treating adjacent A frames as AA."""
    # A, blank, A is a nearly certain alignment for AA with exactly three frames.
    logits = torch.full((1, 3, 37), -12.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 0, 1] = 12
        logits[0, 1, 0] = 12
        logits[0, 2, 1] = 12
    padded_targets = torch.tensor([encode_plate("AA", 3)])
    loss = ctc_batch_loss(logits, padded_targets, ["AA"])
    assert torch.isfinite(loss)
    assert loss.item() < 0.001
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_impossible_repeat_alignment_fails_instead_of_zeroing_the_loss():
    """Silencing impossible labels with zero_infinity would skip supervision invisibly."""
    # Two CTC frames cannot emit AA because its repeated symbols require a separating blank.
    logits = torch.zeros(1, 2, 37)
    with pytest.raises(ValueError, match="frames|alignment"):
        ctc_batch_loss(logits, torch.tensor([encode_plate("AA", 3)]), ["AA"])


def test_ctc_loss_sums_paths_and_normalizes_each_sample_by_target_length():
    """Uniform frame evidence has a hand-countable number of paths for A versus AA."""
    # Three frames emit A through six contiguous runs, but AA only through A,blank,A.
    logits = torch.zeros(2, 3, 37)
    padded_targets = torch.tensor([encode_plate("A", 3), encode_plate("AA", 3)])
    expected_loss = ((3 * math.log(37) - math.log(6)) + 3 * math.log(37) / 2) / 2
    assert ctc_batch_loss(logits, padded_targets, ["A", "AA"]).item() == pytest.approx(
        expected_loss, rel=1e-6
    )


def test_recurrent_encoder_receives_finite_ctc_gradients():
    """The actual model must train through shared frame features and recurrent context."""
    # Two different labels check batch alignment while keeping this a single CPU backward pass.
    torch.set_num_threads(1)
    model = PlateCTC(max_length=3).train()
    images = torch.zeros(2, 3, 64, 192)
    padded_targets = torch.tensor([encode_plate("AA", 3), encode_plate("B", 3)])
    loss = ctc_batch_loss(model(images), padded_targets, ["AA", "B"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.all(torch.isfinite(model.classifier.weight.grad))
    assert model.resnet.conv1.weight.grad is not None
    assert torch.all(torch.isfinite(model.resnet.conv1.weight.grad))


def test_misaligned_or_internal_padding_targets_are_rejected():
    """A padded batch must still encode exactly the text supplied to its CTC loss."""
    # An internal blank is not a legal character target and must not be silently deleted.
    with pytest.raises(ValueError, match="target|padding|label"):
        prepare_ctc_targets(torch.tensor([[1, 0, 1]]), ["AA"], time_steps=5)
    with pytest.raises(ValueError, match="target|label"):
        prepare_ctc_targets(torch.tensor([[2, 2, 0]]), ["AA"], time_steps=5)


def test_ctc_checkpoint_roundtrip_preserves_logits_and_sequence_kind(tmp_path):
    """The generic loader must not reinterpret CTC blank outputs as positional padding."""
    # The shared checkpoint boundary records the model kind before reconstructing its tensors.
    model = PlateCTC(max_length=7).eval()
    image_batch = torch.zeros(1, 3, 64, 192)
    checkpoint_path = tmp_path / "ctc.pt"
    with torch.inference_mode():
        expected_logits = model(image_batch)
    save_checkpoint(model, checkpoint_path, temperature=1.3, metadata={"seed": 42})
    loaded_model, metadata = load_checkpoint(checkpoint_path)
    with torch.inference_mode():
        assert torch.equal(expected_logits, loaded_model(image_batch))
    assert metadata["model_kind"] == "ctc"
    assert metadata["temperature"] == 1.3
    assert isinstance(loaded_model, PlateCTC)


def test_training_output_cannot_replace_its_manifest(tmp_path):
    """The CTC command needs the same source-ownership guard as positional training."""
    # This is intentionally not valid CSV; alias validation must run before manifest parsing.
    manifest_path = tmp_path / "labels.csv"
    manifest_path.write_text("trusted labels\n")
    arguments = Namespace(
        epochs=1,
        batch_size=1,
        threads=1,
        learning_rate=0.0001,
        manifest=manifest_path,
        output=manifest_path,
        seed=42,
    )
    with pytest.raises(ValueError, match="alias|input"):
        run_training(arguments)
    assert manifest_path.read_text() == "trusted labels\n"


def test_training_preflights_all_validation_repeats_before_any_epoch(tmp_path):
    """An impossible late validation label must fail before a partial training run is saved."""
    # Placeholder images deliberately cannot decode; all-record alignment validation must run first.
    manifest_path = tmp_path / "labels.csv"
    (tmp_path / "train.png").write_bytes(b"original train crop")
    (tmp_path / "val.png").write_bytes(b"original validation crop")
    manifest_path.write_text(
        "path,plate,group,split\ntrain.png,AB,train-source,train\nval.png,AAAAAAAAAAAAA,val-source,val\n"
    )
    checkpoint_path = tmp_path / "recognizer.pt"
    arguments = Namespace(
        epochs=1,
        batch_size=1,
        threads=1,
        learning_rate=0.0001,
        manifest=manifest_path,
        output=checkpoint_path,
        seed=42,
        max_length=32,
        device="cpu",
        pretrained=False,
    )
    with pytest.raises(ValueError, match="frames"):
        run_training(arguments)
    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".training.json").exists()
