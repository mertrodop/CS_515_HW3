import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    """Two-layer CNN for MNIST (28×28 greyscale input).

    Architecture: Conv(1→20) → Pool → Conv(20→50) → Pool → FC(800→500) → FC(500→C).

    Args:
        norm: Unused normalization argument kept for API compatibility.
        num_classes: Number of output classes. Default: 10.
    """

    def __init__(self, norm, num_classes: int = 10) -> None:
        """Build conv and FC layers.

        Args:
            norm: Unused; kept for interface consistency with other models.
            num_classes: Number of output classes. Default: 10.
        """
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) #format: (in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(20, 50, 5, 1) 
        self.fc1 = nn.Linear(4 * 4 * 50, 500) # assuming input images are 28x28, after two conv+pool layers we get 4x4 feature maps
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Greyscale image tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        # formula for output size: (W - F + 2P) / S + 1, where W=input size, F=filter size, P=padding, S=stride
        x = F.relu(self.conv1(x)) # (28 - 5 + 2*0) / 1 + 1 = 24
        x = F.max_pool2d(x, 2, 2) # (24 - 2) / 2 + 1 = 12
        x = F.relu(self.conv2(x)) # (12 - 5 + 2*0) / 1 + 1 = 8
        x = F.max_pool2d(x, 2, 2) # (8 - 2) / 2 + 1 = 4, so we have 50 feature maps of size 4x4
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """
    3-conv SimpleCNN for CIFAR-10 (32×32 RGB input).
    Uses BatchNorm + Kaiming initialization.
    """
    def __init__(self, num_classes: int = 10) -> None:
        """Build feature extractor and classifier head, then initialize weights.

        Args:
            num_classes: Number of output classes. Default: 10.
        """
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32×32 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal init to Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: RGB image tensor of shape ``(B, 3, 32, 32)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x