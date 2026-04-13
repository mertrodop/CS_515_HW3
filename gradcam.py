"""Grad-CAM: Gradient-weighted Class Activation Mapping.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Usage example::

    cam = GradCAM(model, target_layer=model.layer4[-1])
    heatmap = cam.generate(input_tensor)   # shape (H, W), values in [0, 1]
    cam.remove_hooks()                     # always clean up
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ---------------------------------------------------------------------------
# GradCAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM using PyTorch forward/backward hooks.

    Registers one forward hook and one backward hook on ``target_layer``.
    The forward hook captures the feature map activations; the backward hook
    captures the gradients of the scalar class score w.r.t. those activations.
    The class activation map is computed as::

        cam = ReLU(sum_k alpha_k * A_k)

    where ``alpha_k = mean_{i,j}(dY^c / dA_k^{ij})`` (global-average-pooled
    gradients) and ``A_k`` is the k-th feature-map channel.

    Args:
        model: The neural network.  Should be in eval mode for inference.
        target_layer: The convolutional layer to hook into.  For ResNet-18
            this is typically ``model.layer4[-1]``; for VGG it is
            ``model.features[-3]``.

    Note:
        Always call :meth:`remove_hooks` (or use the context manager) after
        you are done to avoid memory leaks from retained graph references.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Register forward and backward hooks on *target_layer*.

        Args:
            model: The network to explain.  Must be in eval mode when
                :meth:`generate` is called.
            target_layer: The convolutional layer whose activations and
                gradients are used to build the CAM (typically the last
                conv layer before global pooling).
        """
        self.model        = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    # ── Hooks ───────────────────────────────────────────────────────────────

    def _fwd_hook(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        """Store feature-map activations from the forward pass."""
        self._activations = output.detach()

    def _bwd_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Store gradients flowing back through the target layer."""
        self._gradients = grad_output[0].detach()

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Normalized image tensor, shape ``[1, C, H, W]``.
                Must require gradients — do *not* wrap in ``torch.no_grad()``.
            target_class: Class index to explain.  If ``None``, uses the
                predicted (argmax) class.

        Returns:
            Numpy array of shape ``(H, W)`` with values in ``[0, 1]``.
        """
        self.model.eval()

        # Forward — must NOT be inside no_grad() so backward works
        x = input_tensor.clone().detach().requires_grad_(False)
        x = x.clone()

        # We need gradients through the model, so don't use no_grad here
        logits = self.model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Zero any existing gradients, then backprop the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # alpha_k: global-average-pool of gradients over spatial dims
        # gradients: (1, C, h, w)
        gradients  = self._gradients   # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted combination of feature maps + ReLU
        cam = (weights * activations).sum(dim=1, keepdim=True)   # (1, 1, h, w)
        cam = F.relu(cam)

        # Bilinear upsample to input spatial resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()   # (H, W)

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()

        return cam

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks to free memory."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    # ── Context manager support ─────────────────────────────────────────────

    def __enter__(self) -> "GradCAM":
        """Enter the context manager, returning self."""
        return self

    def __exit__(self, *args) -> None:
        """Exit the context manager and remove all hooks."""
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Helpers: denormalize, overlay, figure
# ---------------------------------------------------------------------------

def _denorm(
    tensor: torch.Tensor,
    mean: tuple = _CIFAR10_MEAN,
    std: tuple  = _CIFAR10_STD,
) -> np.ndarray:
    """Convert a normalized CHW tensor to an HWC uint8 numpy array."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    img = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _overlay_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a Grad-CAM heatmap (red–blue) on an image with transparency.

    Args:
        img_uint8: HWC uint8 image array.
        heatmap: HW float array with values in [0, 1].
        alpha: Heatmap opacity (0 = invisible, 1 = fully opaque).

    Returns:
        HWC uint8 blended image.
    """
    colormap = cm.get_cmap('jet')
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap_rgb + (1 - alpha) * img_uint8).astype(np.uint8)
    return blended


def visualize_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    samples: list,
    output_dir: str = '.',
    tag: str = 'standard',
    n_samples: int = 2,
    device: torch.device = torch.device('cpu'),
    mean: tuple = _CIFAR10_MEAN,
    std: tuple  = _CIFAR10_STD,
) -> None:
    """Generate and save Grad-CAM side-by-side figures.

    For each sample in ``samples`` (up to ``n_samples``) where the model
    classified the clean image correctly but the adversarial image wrongly,
    produce a 4-panel figure::

        [Clean image] [Grad-CAM on clean] [Adv image] [Grad-CAM on adv]

    The Grad-CAM heatmap is overlaid with 50 % opacity using the 'jet'
    colormap (blue = low activation, red = high activation).

    Files are saved as ``{output_dir}/gradcam_sample_{i}_{tag}.png``.

    Args:
        model: Trained model.
        target_layer: Layer to hook for Grad-CAM (e.g. ``model.layer4[-1]``).
        samples: List of ``(clean_img, adv_img, true_lbl, clean_pred, adv_pred)``
            tuples (CPU tensors, single images).  Only samples where
            ``clean_pred == true_lbl`` and ``adv_pred != true_lbl`` are used.
        output_dir: Directory to save PNG files.
        tag: String appended to output filenames (e.g. ``'standard'`` or
            ``'augmix'``).
        n_samples: Maximum number of figures to produce.
        device: Computation device.
        mean: CIFAR-10 per-channel mean.
        std: CIFAR-10 per-channel std.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Filter to misclassification events only
    valid = [s for s in samples if s[3] == s[2] and s[4] != s[2]]
    if not valid:
        print(f'[Grad-CAM] No valid misclassified samples for tag={tag}.')
        return

    cam = GradCAM(model, target_layer)
    count = 0

    for i, (clean_img, adv_img, true_lbl, clean_pred, adv_pred) in enumerate(valid):
        if count >= n_samples:
            break

        clean_t = clean_img.unsqueeze(0).to(device)
        adv_t   = adv_img.unsqueeze(0).to(device)

        heatmap_clean = cam.generate(clean_t, target_class=clean_pred)
        heatmap_adv   = cam.generate(adv_t,   target_class=adv_pred)

        img_clean = _denorm(clean_img, mean, std)
        img_adv   = _denorm(adv_img,   mean, std)

        overlay_clean = _overlay_heatmap(img_clean, heatmap_clean)
        overlay_adv   = _overlay_heatmap(img_adv,   heatmap_adv)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        panels = [
            (img_clean,    f'Clean\npred: {CIFAR10_CLASSES[clean_pred]}'),
            (overlay_clean, f'Grad-CAM (clean)\ntrue: {CIFAR10_CLASSES[true_lbl]}'),
            (img_adv,       f'Adversarial\npred: {CIFAR10_CLASSES[adv_pred]}'),
            (overlay_adv,   f'Grad-CAM (adv)\ntrue: {CIFAR10_CLASSES[true_lbl]}'),
        ]
        for ax, (panel_img, title) in zip(axes, panels):
            ax.imshow(panel_img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        fig.suptitle(
            f'Grad-CAM — {tag} model  |  true: {CIFAR10_CLASSES[true_lbl]}',
            fontsize=12, fontweight='bold',
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f'gradcam_sample_{count}_{tag}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {path}')
        count += 1

    cam.remove_hooks()
    print(f'[Grad-CAM] Done — {count} figure(s) saved for tag={tag}.')
