"""VGG convolutional neural network architectures for image classification."""

from typing import Callable, Union

import torch
import torch.nn as nn


class VGG(nn.Module):
    """VGG network with configurable depth and optional batch normalisation.

    Supports VGG-11, VGG-13, VGG-16, and VGG-19 configurations as defined in
    Simonyan & Zisserman (2014).  The classifier head uses two fully-connected
    layers of width 4096 followed by the output layer.

    Args:
        depth: VGG variant to build.  Must be one of ``'11'``, ``'13'``,
            ``'16'``, or ``'19'``.
        norm: Normalisation layer constructor applied after each convolution
            (default: :class:`torch.nn.BatchNorm2d`).
        num_class: Number of output classes (default: 10).
    """

    def __init__(
        self,
        depth: str,
        norm: Callable[..., nn.Module] = nn.BatchNorm2d,
        num_class: int = 10,
    ) -> None:
        """Build VGG feature extractor and classifier for the given depth.

        Args:
            depth: VGG variant — ``'11'``, ``'13'``, ``'16'``, or ``'19'``.
            norm: Normalization layer constructor. Default: ``nn.BatchNorm2d``.
            num_class: Number of output classes. Default: 10.
        """
        super().__init__()
        self.features = self.make_layers_vgg(depth, norm)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the VGG network.

        Args:
            x: Input image tensor of shape ``(B, C, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_class)``.
        """
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

    def make_layers_vgg(
        self,
        depth: str,
        norm: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> nn.Sequential:
        """Build the convolutional feature extractor for the requested VGG depth.

        Args:
            depth: VGG variant identifier (``'11'``, ``'13'``, ``'16'``, or
                ``'19'``).
            norm: Normalisation layer constructor inserted after every
                convolution (default: :class:`torch.nn.BatchNorm2d`).

        Returns:
            Sequential module containing all convolutional blocks and max-pool
            layers.
        """
        layers: list[nn.Module] = []
        in_channels = 3
        cfg: dict[str, list[Union[int, str]]] = {
            '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        for v in cfg[depth]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, norm(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
