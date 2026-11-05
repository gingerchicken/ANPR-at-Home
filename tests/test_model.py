"""Behavioral checks for plate geometry, checkpoint identity and optional masks."""

import pytest
import torch
from PIL import Image

from anpr.alphabet import encode_plate
from anpr.model import PlateResNet, load_checkpoint, prepare_image, prepare_mask, save_checkpoint


def test_labels_reject_truncation_and_preserve_repeated_characters():
    """Repeated characters are separate positions, unlike CTC collapse semantics."""
    assert encode_plate("AA-11", 5) == [1, 1, 28, 28, 0]
    with pytest.raises(ValueError, match="capacity"):
        # A too-small head must fail instead of silently changing a registration.
        encode_plate("AB12CDE", 3)


def test_unicode_case_expansion_cannot_manufacture_an_ascii_registration():
    """A glyph outside the trained alphabet must fail before uppercase normalization."""
    with pytest.raises(ValueError, match="Unsupported"):
        # Sharp-s uppercases to SS, which would otherwise silently create a different label.
        encode_plate("AB\u00df12", 7)


def test_rgb_and_mask_share_letterbox_geometry():
    """A nonstandard aspect ratio must preserve shape and target alignment."""
    # Both sources fill the same wide rectangle, exposing any mask/resize mismatch.
    image = Image.new("RGB", (400, 40), "white")
    mask = Image.new("L", image.size, 255)
    prepared_image = prepare_image(image)
    prepared_mask = prepare_mask(mask)
    assert prepared_image.shape == (3, 64, 192)
    assert prepared_mask.shape == (1, 64, 192)
    assert prepared_mask[0, 32, 96] == 1
    assert prepared_mask[0, 0, 96] == 0


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("head_layout", ["global", "spatial"])
def test_checkpoint_roundtrip_preserves_logits_and_configuration(tmp_path, use_mask, head_layout):
    """Loading must reconstruct alphabet, capacity, layout, preprocessing and mask mode."""
    # One CPU thread avoids tiny test batches contending across host cores.
    torch.set_num_threads(1)
    model = PlateResNet(max_length=3, use_mask=use_mask, head_layout=head_layout).eval()
    image_batch = torch.zeros(1, 3, 64, 192)
    checkpoint_path = tmp_path / "recognizer.pt"
    with torch.inference_mode():
        # Identical logits after loading prove both architecture and weights survive.
        expected_logits = model(image_batch)
    save_checkpoint(model, checkpoint_path, temperature=1.7, metadata={"source": "test"})
    loaded_model, metadata = load_checkpoint(checkpoint_path)
    with torch.inference_mode():
        assert torch.equal(expected_logits, loaded_model(image_batch))
    assert expected_logits.shape == (1, 3, 37)
    assert metadata["temperature"] == 1.7
    assert loaded_model.use_mask == use_mask
    assert loaded_model.head_layout == head_layout


def test_spatial_heads_keep_each_position_separate():
    """Moving one pooled region must not change another positional head's input."""

    # This backbone isolates the routing contract from convolutional receptive-field overlap.
    class KnownFeatureBackbone(torch.nn.Module):
        """Provide three controlled spatial regions through torchvision's flatten contract."""

        def __init__(self):
            """Keep two feature columns per position so adaptive pooling remains disjoint."""
            super().__init__()
            # Channel zero contains a distinct scalar for each position; other features are zero.
            self.register_buffer("feature_map", torch.zeros(1, 512, 2, 6))
            self.feature_map[:, 0] = torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float32)
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 3))

        def forward(self, images):
            """Preserve the channel-major flattened order used by the real ResNet."""
            return self.avgpool(self.feature_map.expand(images.shape[0], -1, -1, -1)).flatten(1)

    # Each head reads only feature channel zero, making positional leakage directly observable.
    model = PlateResNet(max_length=3, head_layout="spatial").eval()
    model.resnet = KnownFeatureBackbone()
    with torch.no_grad():
        # Every classifier has the same transparent channel-to-class mapping.
        for head in model.fcs:
            head.weight.zero_()
            head.bias.zero_()
            head.weight[1, 0] = 1
        # One black image supplies the batch shape; the controlled backbone owns its features.
        image_batch = torch.zeros(1, 3, 64, 192)
        original_logits = model(image_batch)
        model.resnet.feature_map[:, 0, :, :2] += 5
        changed_logits = model(image_batch)
    assert torch.equal(original_logits[0, :, 1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(changed_logits[0, :, 1], torch.tensor([6.0, 2.0, 3.0]))
    assert torch.equal(original_logits[:, 1:], changed_logits[:, 1:])


def test_spatial_layout_preserves_width_without_new_parameters():
    """The experiment changes spatial processing while retaining matched trainable tensors."""
    # Equal seeds make parameter equality an exact requirement rather than a statistical claim.
    torch.manual_seed(17)
    global_model = PlateResNet(max_length=3)
    torch.manual_seed(17)
    spatial_model = PlateResNet(max_length=3, head_layout="spatial")
    assert global_model.state_dict().keys() == spatial_model.state_dict().keys()
    # Backbone and head weights must start identically before measuring the layout change.
    for parameter_name, global_tensor in global_model.state_dict().items():
        assert torch.equal(global_tensor, spatial_model.state_dict()[parameter_name]), parameter_name
    # Both residual paths retain vertical reduction while avoiding late horizontal downsampling.
    for residual_stage in (spatial_model.resnet.layer3, spatial_model.resnet.layer4):
        assert residual_stage[0].conv1.stride == (2, 1)
        assert residual_stage[0].downsample[0].stride == (2, 1)
    assert spatial_model.resnet.avgpool.output_size == (1, 3)
    # Observe the real encoder input to pooling so stride configuration also has a behavioral check.
    incoming_feature_shapes = []

    def capture_pool_input(_pool, incoming_tensors):
        """Record feature geometry before adaptive pooling can hide width reduction."""
        incoming_feature_shapes.append(tuple(incoming_tensors[0].shape))

    # Remove the temporary observer even if a forward failure interrupts the test.
    pool_observer = spatial_model.resnet.avgpool.register_forward_pre_hook(capture_pool_input)
    try:
        with torch.inference_mode():
            # Spatial stride changes preserve 24 width cells instead of the baseline's six.
            spatial_model.eval()(torch.zeros(1, 3, 64, 192))
    finally:
        pool_observer.remove()
    assert incoming_feature_shapes == [(1, 512, 2, 24)]


def test_mask_experiment_keeps_shared_initialization_and_rng_progression_matched():
    """A U-Net ablation must not also change classifier initialization or caller randomness."""
    # Compare optional-mask construction against exactly the same initial CPU generator state.
    torch.manual_seed(23)
    rgb_model = PlateResNet(max_length=3)
    rgb_next_random = torch.rand(8)
    torch.manual_seed(23)
    masked_model = PlateResNet(max_length=3, use_mask=True)
    masked_next_random = torch.rand(8)
    # The extra channel starts silent while its RGB channels retain the baseline initialization.
    assert torch.equal(rgb_model.resnet.conv1.weight, masked_model.resnet.conv1.weight[:, :3])
    assert torch.count_nonzero(masked_model.resnet.conv1.weight[:, 3:]) == 0
    for parameter_name, rgb_tensor in rgb_model.state_dict().items():
        if parameter_name != "resnet.conv1.weight":
            # All remaining shared tensors, including positional heads, must match exactly.
            assert torch.equal(rgb_tensor, masked_model.state_dict()[parameter_name]), parameter_name
    assert torch.equal(rgb_next_random, masked_next_random)


def test_old_versioned_checkpoint_defaults_to_global_layout(tmp_path):
    """Existing version-one checkpoints predate layout metadata and retain global pooling."""
    # Remove only the new field to reproduce an otherwise valid historical checkpoint.
    checkpoint_path = tmp_path / "old-versioned.pt"
    original_model = PlateResNet(max_length=3).eval()
    save_checkpoint(original_model, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["metadata"].pop("head_layout", None)
    torch.save(checkpoint, checkpoint_path)
    loaded_model, _ = load_checkpoint(checkpoint_path)
    assert loaded_model.head_layout == "global"
    # Strict loading keeps the historical tensors intact after the compatibility default.
    for parameter_name, original_tensor in original_model.state_dict().items():
        assert torch.equal(original_tensor, loaded_model.state_dict()[parameter_name]), parameter_name


def test_checkpoint_layout_is_not_overridden_by_caller_metadata(tmp_path):
    """Architecture metadata must describe the saved weights and processing, not caller notes."""
    # Mislabelled spatial tensors have compatible shapes, making explicit metadata precedence essential.
    checkpoint_path = tmp_path / "spatial.pt"
    model = PlateResNet(max_length=3, head_layout="spatial")
    save_checkpoint(model, checkpoint_path, metadata={"head_layout": "global"})
    loaded_model, metadata = load_checkpoint(checkpoint_path)
    assert metadata["head_layout"] == "spatial"
    assert loaded_model.head_layout == "spatial"


def test_unknown_head_layout_is_rejected():
    """A misspelled layout must fail instead of silently selecting a different architecture."""
    with pytest.raises(ValueError, match="head_layout"):
        PlateResNet(head_layout="unknown")


def test_old_bare_state_dict_requires_explicit_migration(tmp_path):
    """Legacy weights lack the preprocessing metadata required for honest inference."""
    # A bare notebook export cannot identify its geometry or output vocabulary safely.
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save({"resnet.fc.weight": torch.zeros(1)}, checkpoint_path)
    with pytest.raises(ValueError, match="checkpoint|legacy|metadata"):
        load_checkpoint(checkpoint_path)


def test_version_two_requires_explicit_ctc_model_semantics(tmp_path):
    """A new sequence format must not be interpreted as a legacy positional checkpoint."""
    # Positional tensors are deliberately shape-valid, so metadata validation must catch the ambiguity.
    checkpoint_path = tmp_path / "ambiguous.pt"
    save_checkpoint(PlateResNet(max_length=2), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    checkpoint["format_version"] = 2
    checkpoint["metadata"].pop("model_kind", None)
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="model kind|format"):
        load_checkpoint(checkpoint_path)
