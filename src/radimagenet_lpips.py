"""LPIPS variant using a RadImageNet-pretrained ResNet50 backbone.

The backbone is frozen and loaded from a local .pt file. Features are
extracted after the stem (64ch) and after each of the four residual stages
(256, 512, 1024, 2048ch). The uncalibrated mode averages normalised feature
differences across stages without learned lin-layers.

Importing this module registers RadImageNetLPIPS in pyiqa's ARCH_REGISTRY
and injects its entry into DEFAULT_CONFIGS so pyiqa.create_metric works.
"""

import torch
import torch.nn as nn
from torchvision import models

from pyiqa.utils.registry import ARCH_REGISTRY
from pyiqa.archs.arch_util import load_pretrained_network
from pyiqa.archs.lpips_arch import normalize_tensor, spatial_average, NetLinLayer
from pyiqa.default_model_configs import DEFAULT_CONFIGS


_BACKBONE_KEY_REMAP = {
    "backbone.0.": "stem.0.",
    "backbone.1.": "stem.1.",
    "backbone.4.": "layer1.",
    "backbone.5.": "layer2.",
    "backbone.6.": "layer3.",
    "backbone.7.": "layer4.",
}


def _remap_backbone_keys(raw_state: dict) -> dict:
    remapped = {}
    for old_key, value in raw_state.items():
        for old_prefix, new_prefix in _BACKBONE_KEY_REMAP.items():
            if old_key.startswith(old_prefix):
                remapped[old_key.replace(old_prefix, new_prefix, 1)] = value
                break
    return remapped


class _RadImageNetBackbone(nn.Module):
    """Frozen ResNet50 feature extractor with RadImageNet weights.

    Returns intermediate activations after the stem and each residual stage.
    """

    CHANNEL_DIMS = [64, 256, 512, 1024, 2048]

    def __init__(self, weights_path: str):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool   = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        raw_state = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.load_state_dict(_remap_backbone_keys(raw_state), strict=True)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stem_feats   = self.stem(x)
        pooled       = self.pool(stem_feats)
        layer1_feats = self.layer1(pooled)
        layer2_feats = self.layer2(layer1_feats)
        layer3_feats = self.layer3(layer2_feats)
        layer4_feats = self.layer4(layer3_feats)
        return [stem_feats, layer1_feats, layer2_feats, layer3_feats, layer4_feats]


@ARCH_REGISTRY.register()
class RadImageNetLPIPS(nn.Module):
    """LPIPS-style perceptual metric backed by a RadImageNet ResNet50.

    Args:
        backbone_path: Path to the RadImageNet ResNet50.pt weights file.
        calibrated: If True, per-stage NetLinLayer weights are applied.
            Requires pretrained_model_path pointing to trained lin-layer weights.
        pretrained_model_path: Path to calibrated lin-layer weights (.pth).
        use_dropout: Use dropout in lin-layers (only relevant when calibrated).
    """

    def __init__(
        self,
        backbone_path: str,
        calibrated: bool = False,
        pretrained_model_path: str = None,
        use_dropout: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.net = _RadImageNetBackbone(backbone_path)
        self.chns = _RadImageNetBackbone.CHANNEL_DIMS
        self.calibrated = calibrated

        if calibrated:
            self.lins = nn.ModuleList(
                [NetLinLayer(c, use_dropout=use_dropout) for c in self.chns]
            )
            if pretrained_model_path is not None:
                load_pretrained_network(self, pretrained_model_path, strict=False)

        self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual distance between x and y.

        Args:
            x: distorted image tensor (N, 3, H, W) in [0, 1].
            y: reference image tensor (N, 3, H, W) in [0, 1].

        Returns:
            Per-sample distance scores (N,). Lower = more similar.
        """
        def normalize_and_extract(img):
            return self.net((img - self.mean) / self.std)

        distorted_feats  = normalize_and_extract(x)
        reference_feats  = normalize_and_extract(y)

        stage_scores = []
        for k in range(len(self.chns)):
            normalized_diff = (
                normalize_tensor(distorted_feats[k]) - normalize_tensor(reference_feats[k])
            ) ** 2
            if self.calibrated:
                stage_scores.append(spatial_average(self.lins[k](normalized_diff), keepdim=True))
            else:
                stage_scores.append(spatial_average(normalized_diff.sum(dim=1, keepdim=True), keepdim=True))

        return sum(stage_scores).squeeze(-1).squeeze(-1)


DEFAULT_CONFIGS["radimagenet_lpips"] = {
    "metric_opts": {"type": "RadImageNetLPIPS"},
    "metric_mode": "FR",
    "lower_better": True,
    "score_range": "0, 1",
}
