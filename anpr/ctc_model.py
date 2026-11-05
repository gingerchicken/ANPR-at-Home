"""Width-preserving ResNet18 and recurrent CTC plate-recognition experiment."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from anpr.alphabet import NUM_CLASSES
from anpr.model import IMAGE_SIZE, preserve_horizontal_resolution


class PlateCTC(nn.Module):
    """Return frame logits for variable-length plates without fixed character slots.

    Inputs follow the shared normalized RGB BCHW contract. Outputs are B,T,37;
    class zero is a CTC blank, while repeated target symbols remain distinct.
    The default 192-pixel input width supplies 24 frames. Max length limits
    registration decoding and labels, not the number of model output frames.
    """

    def __init__(self, max_length: int = 7, pretrained: bool = False):
        """Reuse ImageNet kernels, retain width and add one bidirectional recurrent layer."""
        super().__init__()
        if isinstance(max_length, bool) or not isinstance(max_length, int) or not 1 <= max_length <= 32:
            # Capacity mistakes must fail before allocating a model or downloading initialization.
            raise ValueError("max_length must be an integer between 1 and 32")
        # These attributes define checkpoint/decoder semantics rather than guessing from tensor names.
        self.max_length = max_length
        self.model_kind = "ctc"
        self.use_mask = False
        self.head_layout = "ctc"
        # Horizontal reduction ends at stride eight; ceil division matches the padded convolution stages.
        self.time_steps = (IMAGE_SIZE[0] + 7) // 8
        # The encoder keeps the positional experiment's exact ResNet18 feature kernels.
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        preserve_horizontal_resolution(self.resnet)
        # Pool only vertically so torchvision can retain ownership of the backbone forward path.
        feature_count = self.resnet.fc.in_features
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.resnet.fc = nn.Identity()
        # Shared recurrent and classification weights allow character alignment to vary with plate length.
        self.sequence_model = nn.LSTM(feature_count, 128, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one shared-vocabulary logit vector per retained horizontal frame."""
        if images.ndim != 4 or images.shape[1] != 3:
            # Reject mask-only or grayscale batches before the RGB feature extractor consumes them.
            raise ValueError("Expected normalized BCHW images with three RGB channels")
        # Torchvision flattens B,C,1,T; restore channels before making time the sequence axis.
        flattened_features = self.resnet(images)
        frame_features = flattened_features.reshape(images.shape[0], 512, -1).transpose(1, 2)
        # Final recurrent states are discarded so each crop starts independently of earlier plates.
        contextual_features, _ = self.sequence_model(frame_features)
        return self.classifier(contextual_features)
