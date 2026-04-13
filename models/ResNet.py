import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """Wraps an arbitrary callable as an ``nn.Module``.

    Used to apply zero-padding shortcuts (option A) in ResNet without
    defining a separate named class.

    Args:
        lambd: Any callable that maps a tensor to a tensor.
    """

    def __init__(self, lambd) -> None:
        """Store the callable.

        Args:
            lambd: Callable applied to the input tensor in :meth:`forward`.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped callable to *x*.

        Args:
            x: Input tensor.

        Returns:
            Output tensor produced by ``self.lambd(x)``.
        """
        return self.lambd(x)

class BasicBlock(nn.Module):
    """
    Basic residual block used in shallow ResNet architectures (e.g., ResNet-18, ResNet-34).

    This block consists of two 3×3 convolutional layers with Batch Normalization
    and ReLU activation. A residual (skip) connection adds the input tensor to
    the output of the stacked convolutions.

    If the spatial resolution or number of channels changes (due to stride > 1
    or channel mismatch), a projection shortcut (typically a 1×1 convolution)
    is applied to the input to match dimensions before addition.

    Structure:
        Conv3x3(in_channels → channels, stride)
        BatchNorm
        ReLU
        Conv3x3(channels → channels, stride=1)
        BatchNorm
        Add shortcut
        ReLU

    Args:
        in_channels (int):
            Number of input channels.
        channels (int):
            Number of output channels produced by the block.
        stride (int, optional):
            Stride for the first convolution layer. Default is 1.
        downsample (nn.Module, optional):
            Optional module to match spatial or channel dimensions
            for the residual connection.

    Attributes:
        expansion (int):
            Expansion factor for output channels. For BasicBlock, expansion = 1.

    Shape:
        Input:
            (N, in_channels, H, W)
        Output:
            (N, channels * expansion, H_out, W_out)

        where:
            H_out = H / stride
            W_out = W / stride
    """
  
    expansion = 1 # For BasicBlock, output channels = channels * expansion = channels
    def __init__(self, in_channels: int, channels: int, stride: int = 1, norm=nn.BatchNorm2d, option: str = 'B') -> None:
        """Build the two-conv residual block with optional projection shortcut.

        Args:
            in_channels: Number of input channels.
            channels: Number of output channels.
            stride: Stride for the first convolution. Default: 1.
            norm: Normalization layer constructor. Default: ``nn.BatchNorm2d``.
            option: Shortcut type — ``'A'`` for zero-padding, ``'B'`` for 1×1 conv projection.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, channels//4, channels//4), "constant", 0))
                # The slicing x[:, :, ::2, ::2] performs downsampling by taking every second pixel in height and width dimensions.
                # (1, 16, 32, 32) → (1, 16, 16, 16) after slicing.
                # Format: (padding_left, padding_right, padding_top, padding_bottom, padding_channel_left, padding_channel_right).
                # (1, 16, 16, 16) → (1, 32, 16, 16) after padding channels from 16 to 32, channels//4 = 32//4 = 8 zeros added to the left and right of the channel dimension
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, self.expansion * channels, kernel_size=1, stride=stride, bias=False),
                     norm(self.expansion * channels)
                )
                # (1, 16, 32, 32) → (1, 32, 16, 16) after 1×1 convolution with stride=2 and output channels=32.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to *x*.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(B, channels, H_out, W_out)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet (Residual Network) implementation for image classification.

    This class builds a ResNet architecture using either BasicBlock or Bottleneck
    residual blocks. The network consists of an initial convolution and
    normalization, followed by four residual layers, global average pooling,
    and a final fully connected layer for classification.

    Args:
    -----
        block (nn.Module):
            Residual block class to use (BasicBlock or Bottleneck).
        num_blocks (list of int):
            Number of blocks in each of the four layers.
            Example: [2, 2, 2, 2] for ResNet-18.
        norm (nn.Module, optional):
            Normalization layer to use after convolutions. Default: nn.BatchNorm2d.
        num_classes (int, optional):
            Number of output classes for classification. Default: 10.

    Attributes:
    -----------
        in_channels (int):
            Number of input channels for the next block; updated after each layer.
        conv1 (nn.Conv2d):
            Initial convolution layer (3×3 kernel).
        bn1 (nn.Module):
            Normalization after conv1.
        layer1, layer2, layer3, layer4 (nn.Sequential):
            Residual layers composed of the specified block type.
        avgpool (nn.AdaptiveAvgPool2d):
            Global average pooling layer reducing spatial dimensions to 1×1.
        linear (nn.Linear):
            Fully connected layer mapping features to `num_classes`.

    Shape:
        Input:
            (N, 3, H, W) where H and W are image height and width.
        Output:
            (N, num_classes) — class logits for each input sample.

    Example:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> x = torch.randn(8, 3, 32, 32)
        >>> logits = model(x)

    References:
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    [2] https://github.com/KaimingHe/deep-residual-networks
    
    """
    def __init__(self, block, num_blocks: list, norm=nn.BatchNorm2d, num_classes: int = 10) -> None:
        """Build all residual stages and the classification head.

        Args:
            block: Residual block class (e.g. ``BasicBlock``).
            num_blocks: Number of blocks per stage, e.g. ``[2, 2, 2, 2]``.
            norm: Normalization layer constructor. Default: ``nn.BatchNorm2d``.
            num_classes: Number of output classes. Default: 10.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm=norm,stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm=norm,stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm=norm,stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm=norm,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, channels: int, num_blocks: int, norm: nn.Module, stride: int) -> nn.Sequential:
        """Build one residual stage consisting of *num_blocks* blocks.

        Args:
            block: Residual block class (e.g. ``BasicBlock``).
            channels: Number of output channels for every block in this stage.
            num_blocks: Number of blocks to stack.
            norm: Normalization layer constructor passed to each block.
            stride: Stride for the first block; subsequent blocks use stride 1.

        Returns:
            ``nn.Sequential`` containing all blocks for the stage.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride,norm))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the full ResNet.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out