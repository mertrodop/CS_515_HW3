"""Multilayer Perceptron (MLP) architectures for image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multilayer perceptron with BatchNorm, ReLU, and Dropout after each hidden layer.

    Flattens the input before passing it through a configurable stack of
    fully-connected blocks, each consisting of Linear → BatchNorm1d → ReLU →
    Dropout, followed by a final linear output layer.

    Args:
        input_size: Number of input features (e.g. 784 for 28×28 MNIST images).
        hidden_sizes: Sequence of hidden layer widths.
        num_classes: Number of output classes.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        """Build the stacked FC blocks and output layer.

        Args:
            input_size: Number of input features.
            hidden_sizes: Width of each hidden layer.
            num_classes: Number of output classes.
            dropout: Dropout probability after each hidden layer.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape ``(B, *)``.  Flattened internally to
               ``(B, input_size)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = x.view(x.size(0), -1)   # flatten (B, 1, 28, 28) → (B, 784)
        return self.net(x)


class MLP2(nn.Module):
    """Lightweight MLP without BatchNorm, using ReLU activations only.

    Flattens the input and applies a sequence of linear layers with ReLU
    activations, followed by a linear output layer producing raw logits.

    Args:
        input_dim: Number of input features (default: 784 for 28×28 images).
        hidden_dims: Sequence of hidden layer widths.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] = [512, 256],
        num_classes: int = 10,
    ) -> None:
        """Build hidden layers and the output layer.

        Args:
            input_dim: Number of input features. Default: 784.
            hidden_dims: Width of each hidden layer.
            num_classes: Number of output classes. Default: 10.
        """
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape ``(B, *)``.  Flattened internally to
               ``(B, input_dim)``.

        Returns:
            Logit tensor of shape ``(B, num_classes)``.
        """
        x = x.view(x.size(0), -1)  # flatten

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)  # logits
        return x
