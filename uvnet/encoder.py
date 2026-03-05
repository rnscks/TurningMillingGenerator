"""
UV-Net Surface/Curve Encoder

- SurfaceEncoder : Face UV-grid (N, 7, uv, uv)  → (N, 64)
- CurveEncoder   : Edge U-grid  (E, 6, uv)       → (E, 64)
"""

import torch
import torch.nn as nn


def _conv2d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True),
    )


def _conv1d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.LeakyReLU(inplace=True),
    )


def _fc(in_f: int, out_f: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_f, out_f, bias=False),
        nn.BatchNorm1d(out_f),
        nn.LeakyReLU(inplace=True),
    )


class SurfaceEncoder(nn.Module):
    """
    Face UV-grid → 64-dim embedding.

    Input  : (N, 7, num_uv, num_uv)
               7 = points(3) + normals(3) + visibility(1)
    Output : (N, 64)
    """

    def __init__(self, num_uv: int = 10):
        super().__init__()
        self.num_uv = num_uv
        self.conv1 = _conv2d(7, 64)
        self.conv2 = _conv2d(64, 128)
        self.conv3 = _conv2d(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = _fc(256, 64)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 7, uv, uv)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)  # (N, 256)
        return self.fc(x)                       # (N, 64)


class CurveEncoder(nn.Module):
    """
    Edge U-grid → 64-dim embedding.

    Input  : (E, 6, num_uv)
               6 = points(3) + tangents(3)
    Output : (E, 64)
    """

    def __init__(self, num_uv: int = 10):
        super().__init__()
        self.num_uv = num_uv
        self.conv1 = _conv1d(6, 64)
        self.conv2 = _conv1d(64, 128)
        self.conv3 = _conv1d(128, 256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = _fc(256, 64)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, 6, uv)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)  # (E, 256)
        return self.fc(x)                       # (E, 64)
