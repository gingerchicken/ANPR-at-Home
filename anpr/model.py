"""ResNet18 plate baseline, optional soft-mask U-Net and explicit checkpoints."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as functional
from torchvision.models import ResNet18_Weights, resnet18

from anpr.alphabet import CHARACTERS, NUM_CLASSES

# Shared geometry is recorded in checkpoints because a preprocessing change needs evaluation.
IMAGE_SIZE = (192, 64)
PREPROCESS_VERSION = "rgb-letterbox-192x64-imagenet-v1"
# ImageNet normalization matches the optional pretrained backbone, not a plate-specific calibration.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _letterbox(image: Image.Image, *, is_mask: bool) -> Image.Image:
    """Apply identical resize/pad geometry to images and aligned character masks."""
    # Masks require nearest interpolation and zero borders; RGB uses neutral gray padding.
    converted_image = image.convert("L" if is_mask else "RGB")
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    return ImageOps.pad(
        converted_image, IMAGE_SIZE, method=resampling, color=0 if is_mask else (127, 127, 127)
    )


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Return normalized float32 RGB CHW pixels while preserving the crop aspect ratio."""
    # Copy PIL storage so tensor writes cannot mutate or reference a read-only image array.
    pixel_array = np.array(_letterbox(image, is_mask=False), dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(pixel_array).permute(2, 0, 1)
    mean_tensor = image_tensor.new_tensor(IMAGE_MEAN)[:, None, None]
    std_tensor = image_tensor.new_tensor(IMAGE_STD)[:, None, None]
    return (image_tensor - mean_tensor) / std_tensor


def prepare_mask(mask: Image.Image) -> torch.Tensor:
    """Return aligned 1HW foreground targets in [0,1], with zero letterbox padding."""
    # Preserve any soft antialiasing in the source target instead of binarizing thin strokes.
    mask_array = np.array(_letterbox(mask, is_mask=True), dtype=np.float32) / 255.0
    return torch.from_numpy(mask_array).unsqueeze(0)


def preserve_horizontal_resolution(backbone: nn.Module) -> None:
    """Retain late ResNet18 width while keeping vertical downsampling and pretrained kernels.

    Both paths in each residual block must use matching strides. This shared
    boundary keeps positional and CTC experiments on the same spatial encoder.
    """
    # Only the first block in each stage downsamples; its shortcut must change with its convolution.
    for residual_stage in (backbone.layer3, backbone.layer4):
        # Horizontal stride one retains 24 feature columns for the standard 192-pixel input.
        first_block = residual_stage[0]
        first_block.conv1.stride = (2, 1)
        first_block.downsample[0].stride = (2, 1)
        first_block.stride = (2, 1)


class _MaskBlock(nn.Module):
    """Two spatial convolutions with normalization independent of training batch size."""

    def __init__(self, input_channels: int, output_channels: int):
        """Keep skip-map detail using padding rather than cropping feature boundaries."""
        super().__init__()
        # Group normalization supports small plate batches without unstable batch statistics.
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Preserve spatial dimensions for the U-Net skip connections."""
        return self.layers(images)


class SoftMaskUNet(nn.Module):
    """Small character/background U-Net used only in the optional mask experiment."""

    def __init__(self):
        """Use two downsampling levels to retain strokes at low plate resolution."""
        super().__init__()
        # Encoder maps and decoder blocks retain original-resolution evidence through skips.
        self.encoder = _MaskBlock(3, 8)
        self.middle = _MaskBlock(8, 16)
        self.bottleneck = _MaskBlock(16, 32)
        self.decoder_middle = _MaskBlock(48, 16)
        self.decoder_output = _MaskBlock(24, 8)
        self.output = nn.Conv2d(8, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return foreground logits, without making irreversible pixel decisions."""
        # Each skip preserves fine boundaries that would otherwise vanish during pooling.
        full_features = self.encoder(images)
        half_features = self.middle(functional.max_pool2d(full_features, 2))
        quarter_features = self.bottleneck(functional.max_pool2d(half_features, 2))
        middle_features = self.decoder_middle(
            torch.cat(
                [
                    functional.interpolate(
                        quarter_features, size=half_features.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    half_features,
                ],
                dim=1,
            )
        )
        output_features = self.decoder_output(
            torch.cat(
                [
                    functional.interpolate(
                        middle_features, size=full_features.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    full_features,
                ],
                dim=1,
            )
        )
        return self.output(output_features)


class PlateResNet(nn.Module):
    """Independent positional character heads over a shared ResNet18 representation.

    Inputs are normalized BCHW RGB plate crops. Outputs are B x max_length x 37
    logits, with class zero reserved for trailing padding. No CTC collapse applies.
    The global layout retains the original pooled baseline. The spatial layout
    retains late horizontal resolution and supplies a separate width region to
    each head, without adding classifier parameters. Its wider activation maps
    increase memory and convolution work and must be judged by measured recall.
    """

    def __init__(
        self,
        max_length: int = 7,
        pretrained: bool = False,
        use_mask: bool = False,
        head_layout: str = "global",
    ):
        """Initialize matched shared weights, with optional spatial heads and a soft mask.

        Global and spatial layouts consume identical initialization randomness.
        Optional mask components use a forked CPU generator so enabling a U-Net
        does not also change the shared classifier weights or caller RNG state.
        Existing trained tensors load strictly regardless of initialization order.
        """
        super().__init__()
        if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= 32:
            # Bound capacity to keep accidental configuration errors from allocating huge heads.
            raise ValueError("max_length must be an integer between 1 and 32")
        if head_layout not in ("global", "spatial"):
            # Layouts share tensor shapes, so a typo must not silently change their semantics.
            raise ValueError("head_layout must be 'global' or 'spatial'")
        # Capacity, mask mode and layout are metadata, not inferred from state-dict tensor shapes.
        self.max_length = max_length
        self.use_mask = use_mask
        self.head_layout = head_layout
        # Model kind distinguishes positional padding semantics from sequence-alignment blanks.
        self.model_kind = "positional"
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Create shared classifiers before optional components so ablations start from the same tensors.
        feature_count = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fcs = nn.ModuleList(nn.Linear(feature_count, NUM_CLASSES) for _ in range(max_length))
        if head_layout == "spatial":
            # Use the same spatial encoder boundary as the CTC sequence experiment.
            preserve_horizontal_resolution(self.resnet)
            # Torchvision flattens this channel-major map; forward restores its positional axis.
            self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, max_length))
        # Keep the global layout unchanged as the measured baseline, rather than replacing it silently.
        self.mask_network = None
        if use_mask:
            with torch.random.fork_rng(devices=[]):
                # Mask-only initialization must not perturb shared weights or future caller randomness.
                self.mask_network = SoftMaskUNet()
                # The RGB convolution supplies shared weights; its replacement adds one silent channel.
                rgb_convolution = self.resnet.conv1
                mask_convolution = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    # Zero the fourth channel initially; recognition learns whether masks are useful.
                    mask_convolution.weight[:, :3].copy_(rgb_convolution.weight)
                    mask_convolution.weight[:, 3:].zero_()
                self.resnet.conv1 = mask_convolution

    def mask_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Expose foreground logits for aligned auxiliary segmentation supervision."""
        if self.mask_network is None:
            # A missing mask model is a configuration error, not an all-background prediction.
            raise ValueError("This checkpoint does not include a soft-mask U-Net")
        return self.mask_network(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return positional logits; retain RGB when the mask experiment is enabled."""
        if images.ndim != 4 or images.shape[1] != 3:
            # Reject grayscale or mask-only batches before they reach the RGB backbone.
            raise ValueError("Expected normalized BCHW images with three RGB channels")
        # Auxiliary mask probabilities add evidence without thresholding or replacing image pixels.
        recognition_images = images
        if self.use_mask:
            recognition_images = torch.cat([images, self.mask_logits(images).sigmoid()], dim=1)
        # Torchvision owns the convolutional forward path for both layouts, avoiding a duplicated encoder.
        features = self.resnet(recognition_images)
        if self.head_layout == "spatial":
            # Flatten order is B,C,1,L, so reshape before transposing rather than interleaving channels.
            positional_features = features.reshape(features.shape[0], -1, self.max_length).transpose(1, 2)
            return torch.stack(
                [head(positional_features[:, position]) for position, head in enumerate(self.fcs)], dim=1
            )
        # The original globally pooled heads remain available for the matched baseline experiment.
        return torch.stack([head(features) for head in self.fcs], dim=1)


def save_checkpoint(
    model: nn.Module, path: str | Path, temperature: float = 1.0, metadata: dict | None = None
) -> None:
    """Save weights with explicit architecture and preprocessing, never a pickled model."""
    if not math.isfinite(temperature) or temperature <= 0:
        # Temperature is a validation-derived scaling parameter, always strictly positive.
        raise ValueError("temperature must be finite and positive")
    # CTC uses a new format so an older positional-only loader cannot misinterpret its tensors.
    model_kind = getattr(model, "model_kind", "positional")
    if model_kind not in ("positional", "ctc"):
        raise ValueError("Unsupported recognition model kind")
    # Reserved metadata wins over caller notes so it cannot misdescribe the saved architecture.
    checkpoint = {
        "format_version": 2 if model_kind == "ctc" else 1,
        "state_dict": model.state_dict(),
        "metadata": {
            **(metadata or {}),
            "max_length": model.max_length,
            "use_mask": model.use_mask,
            "head_layout": model.head_layout,
            "model_kind": model_kind,
            "alphabet": CHARACTERS,
            "preprocess": PREPROCESS_VERSION,
            "temperature": temperature,
        },
    }
    # Create the parent for an explicitly selected output, then replace only after a full write.
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a versioned checkpoint using weights-only deserialization and strict matching.

    Bare notebook exports are rejected because their image preprocessing is absent.
    A loaded checkpoint is placed in evaluation mode and does not download weights.
    """
    # Restrict unpickling to tensor/state metadata rather than executing a saved model object.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") not in (1, 2):
        # A legacy migration must explicitly identify and validate its preprocessing first.
        raise ValueError("Unsupported checkpoint or legacy weights without preprocessing metadata")
    # Vocabulary and geometry identity must hold before the tensor shapes are trusted.
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    # Version one predates model kinds and always denotes the positional recognizer.
    model_kind = metadata.get("model_kind", "positional")
    if (
        model_kind not in ("positional", "ctc")
        or (model_kind == "ctc" and checkpoint["format_version"] != 2)
        or (checkpoint["format_version"] == 2 and model_kind != "ctc")
    ):
        raise ValueError("Checkpoint model kind is incompatible with its format version")
    if metadata.get("alphabet") != CHARACTERS or metadata.get("preprocess") != PREPROCESS_VERSION:
        raise ValueError("Checkpoint alphabet or preprocessing metadata does not match this recognizer")
    # Temperature may remain one if no validation calibration has been performed.
    temperature = metadata.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Checkpoint temperature must be finite and positive")
    # Older version-one exports predate layout metadata and retain their original global-pool behavior.
    metadata = {**metadata, "head_layout": metadata.get("head_layout", "global"), "model_kind": model_kind}
    # No pretrained initialization is needed because all trained tensors must be present.
    if model_kind == "ctc":
        # Lazy import avoids a model-module cycle while sharing preprocessing and checkpoint validation.
        from anpr.ctc_model import PlateCTC

        if metadata.get("use_mask") or metadata.get("head_layout") != "ctc":
            raise ValueError("CTC checkpoint metadata does not match the supported sequence architecture")
        model = PlateCTC(max_length=metadata["max_length"])
    else:
        # Original positional checkpoints retain their exact global/spatial interpretation.
        model = PlateResNet(
            max_length=metadata["max_length"],
            use_mask=metadata.get("use_mask", False),
            head_layout=metadata["head_layout"],
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata
