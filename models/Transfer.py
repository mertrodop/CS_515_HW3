import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision import models


@dataclass
class Transfer_Config:
    """Configuration for transfer learning models.

    Attributes:
        backbone: Pretrained model to use, either "resnet18" or "vgg16".
        num_classes: Number of output classes (10 for CIFAR-10).
        option: Transfer learning strategy (1 = freeze backbone, 2 = modify early convs).
    """
    backbone: str = "resnet18"
    num_classes: int = 10
    option: int = 1


class TransferOption1(nn.Module):
    """Pretrained backbone with frozen weights, only FC layer trained.

    Loads a pretrained ResNet-18 or VGG-16, freezes all backbone layers,
    and replaces the final FC layer for CIFAR-10 classification.
    Expects input images resized to 224x224 to match ImageNet dimensions.

    Args:
        config: Transfer_Config specifying backbone and num_classes.
    """

    def __init__(self, config: Transfer_Config) -> None:
        """Load pretrained backbone, freeze all weights, replace final FC layer.

        Args:
            config: Specifies backbone (``'resnet18'`` or ``'vgg16'``) and
                number of output classes.
        """
        super().__init__()

        if config.backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = self.model.fc.in_features

            for param in self.model.parameters():
                param.requires_grad = False

            self.model.fc = nn.Linear(num_features, config.num_classes)

        elif config.backbone == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            num_features = self.model.classifier[6].in_features

            for param in self.model.parameters():
                param.requires_grad = False

            self.model.classifier[6] = nn.Linear(num_features, config.num_classes)

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        self._freeze_bn()

    def _freeze_bn(self) -> None:
        """Set all BatchNorm layers to eval mode to prevent running stats from updating."""
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> "TransferOption1":
        """Override train() to keep frozen BN layers in eval mode."""
        super().train(mode)
        self._freeze_bn()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass *x* through the frozen backbone and fine-tuned FC head.

        Args:
            x: Image tensor of shape ``(B, 3, 224, 224)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.model(x)


class TransferOption2(nn.Module):
    """Pretrained backbone adapted for 32x32 CIFAR-10 input, all layers trainable.

    Modifies early conv layers to accept 32x32 images directly instead of
    resizing to 224x224. Fine-tunes the entire network including early layers.

    Args:
        config: Transfer_Config specifying backbone and num_classes.
    """

    def __init__(self, config: Transfer_Config) -> None:
        """Load pretrained backbone and adapt it for 32×32 CIFAR-10 input.

        Args:
            config: Specifies backbone (``'resnet18'`` or ``'vgg16'``) and
                number of output classes.
        """
        super().__init__()

        if config.backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
            self.model.fc      = nn.Linear(512, config.num_classes)

        elif config.backbone == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.model.classifier = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, config.num_classes),
            )

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass *x* through the fully fine-tuned backbone.

        Args:
            x: Image tensor of shape ``(B, 3, 32, 32)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        return self.model(x)


def build_transfer_model(config: Transfer_Config) -> nn.Module:
    """Instantiate the correct transfer learning model based on config.option.

    Args:
        config: Transfer_Config with backbone, num_classes, and option fields.

    Returns:
        An nn.Module ready to be moved to a device.
    """
    if config.option == 1:
        return TransferOption1(config)
    if config.option == 2:
        return TransferOption2(config)
    raise ValueError(f"Unknown option: {config.option}")
