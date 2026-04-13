"""AugMix data augmentation and JSD consistency loss.

Implements the AugMix framework from:
    Hendrycks et al., "AugMix: A Simple Data Processing Method to Improve
    Robustness and Uncertainty", ICLR 2020.

Usage:
    from augmix import AugMixConfig, AugMixTransform, JSDLoss, augmix_worker_init_fn
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation operations (PIL → PIL)
# Excluded per spec: contrast, color, brightness, sharpness, cutout
# ---------------------------------------------------------------------------

# Severity magnitude tables, indexed by severity 1-5, tuned for 32x32 images.
_ROTATE_DEGREES    = [0, 15, 20, 25, 30]
_TRANSLATE_PX      = [0, 3,  5,  7,  10]
_SHEAR_DEGREES     = [0, 5,  8,  12, 15]
_POSTERIZE_BITS    = [8, 4,  3,  2,  1]
_SOLARIZE_THRESH   = [256, 224, 192, 128, 64]


def _severity_idx(severity: int) -> int:
    """Clamp severity to valid index range [0, 4]."""
    return max(0, min(severity, 5) - 1)


def autocontrast(img: Image.Image, severity: int) -> Image.Image:
    """Apply autocontrast (severity unused — operation is binary)."""
    return ImageOps.autocontrast(img)


def equalize(img: Image.Image, severity: int) -> Image.Image:
    """Equalize the image histogram (severity unused)."""
    return ImageOps.equalize(img)


def posterize(img: Image.Image, severity: int) -> Image.Image:
    """Reduce the number of bits for each color channel."""
    bits = _POSTERIZE_BITS[_severity_idx(severity)]
    return ImageOps.posterize(img, bits)


def solarize(img: Image.Image, severity: int) -> Image.Image:
    """Invert all pixels above a threshold."""
    threshold = _SOLARIZE_THRESH[_severity_idx(severity)]
    return ImageOps.solarize(img, threshold)


def rotate(img: Image.Image, severity: int) -> Image.Image:
    """Rotate the image by a random angle in [-deg, +deg]."""
    deg = _ROTATE_DEGREES[_severity_idx(severity)]
    angle = random.uniform(-deg, deg)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def shear_x(img: Image.Image, severity: int) -> Image.Image:
    """Apply horizontal shear transformation."""
    import math
    deg = _SHEAR_DEGREES[_severity_idx(severity)]
    shear = math.tan(math.radians(random.uniform(-deg, deg)))
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, shear, -shear * h / 2, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def shear_y(img: Image.Image, severity: int) -> Image.Image:
    """Apply vertical shear transformation."""
    import math
    deg = _SHEAR_DEGREES[_severity_idx(severity)]
    shear = math.tan(math.radians(random.uniform(-deg, deg)))
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, shear, 1, -shear * w / 2),
        resample=Image.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def translate_x(img: Image.Image, severity: int) -> Image.Image:
    """Translate image horizontally by up to N pixels."""
    px = _TRANSLATE_PX[_severity_idx(severity)]
    delta = random.randint(-px, px)
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, delta, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=(128, 128, 128),
    )


def translate_y(img: Image.Image, severity: int) -> Image.Image:
    """Translate image vertically by up to N pixels."""
    px = _TRANSLATE_PX[_severity_idx(severity)]
    delta = random.randint(-px, px)
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, 0, 1, delta),
        resample=Image.BILINEAR,
        fillcolor=(128, 128, 128),
    )


# Ordered list of allowed augmentation operations (contrast/color/brightness/
# sharpness/cutout excluded to avoid overlap with CIFAR-10-C corruptions).
_AUGMENTATIONS: list[Callable[[Image.Image, int], Image.Image]] = [
    autocontrast,
    equalize,
    posterize,
    solarize,
    rotate,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AugMixConfig:
    """Hyperparameters for the AugMix transform.

    Attributes:
        k: Number of augmentation chains to mix (paper default: 3).
        alpha: Concentration parameter for Dirichlet and Beta distributions
               (paper default: 1.0, giving a uniform prior).
        severity: Augmentation operation severity on a scale of 1–5
                  (paper default: 3).
    """
    k: int = 3
    alpha: float = 1.0
    severity: int = 3


# ---------------------------------------------------------------------------
# AugMix transform
# ---------------------------------------------------------------------------

class AugMixTransform:
    """AugMix data augmentation transform for CIFAR-10 (32×32 RGB images).

    Replaces the standard training Compose pipeline.  Each call returns a
    3-tuple of normalized tensors ``(x_orig, x_aug1, x_aug2)`` where
    ``x_orig`` is the clean preprocessed image and ``x_aug1``, ``x_aug2``
    are two independent AugMix samples derived from the same preprocessed
    image.

    The preprocessing order is:
        1. RandomCrop(32, padding=4) + RandomHorizontalFlip  applied **once**
           to produce a preprocessed PIL image.
        2. ``x_orig`` is obtained by ToTensor + Normalize on that PIL image.
        3. For each AugMix sample, k augmentation chains are applied to the
           preprocessed PIL image, mixed with Dirichlet weights, and blended
           with the original via a Beta-distributed skip connection.

    Args:
        mean: Per-channel normalization mean (e.g. CIFAR-10 mean).
        std:  Per-channel normalization std.
        config: :class:`AugMixConfig` controlling k, alpha, severity.
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        config: AugMixConfig | None = None,
    ) -> None:
        self.config = config or AugMixConfig()
        # Pre-augmentation: random spatial transforms applied once per sample.
        self._pre = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        # Tensor conversion pipeline used for both original and augmented views.
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(
        self, img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply AugMix to a PIL image and return three normalized tensors.

        Args:
            img: Input PIL image (RGB, 32×32).

        Returns:
            ``(x_orig, x_aug1, x_aug2)`` each of shape ``[3, 32, 32]``.
        """
        img = img.convert("RGB")
        # Step 1: shared spatial preprocessing (crop + flip).
        img_pre: Image.Image = self._pre(img)
        # Step 2: clean view.
        x_orig: torch.Tensor = self._to_tensor(img_pre)
        # Step 3: two independent AugMix samples.
        x_aug1: torch.Tensor = self._augmix_single(img_pre, x_orig)
        x_aug2: torch.Tensor = self._augmix_single(img_pre, x_orig)
        return x_orig, x_aug1, x_aug2

    def _augmix_single(
        self, img_pre: Image.Image, x_orig: torch.Tensor
    ) -> torch.Tensor:
        """Produce one AugMix sample from a preprocessed PIL image.

        Implements Algorithm 1 of Hendrycks et al., ICLR 2020:
            - Sample mixing weights w ~ Dirichlet([alpha]*k).
            - Sample skip-connection weight m ~ Beta(alpha, alpha).
            - For each chain i: apply d ~ Uniform{1,2,3} random ops in sequence.
            - Mix chains: x_mixed = sum_i w_i * aug_i.
            - Return: m * x_orig + (1-m) * x_mixed.

        Args:
            img_pre: Preprocessed PIL image (after crop/flip, before ToTensor).
            x_orig: Normalized tensor of ``img_pre`` (avoids re-computation).

        Returns:
            AugMix-augmented tensor of shape ``[3, 32, 32]``.
        """
        cfg = self.config
        w: np.ndarray = np.random.dirichlet([cfg.alpha] * cfg.k).astype(np.float32)
        m: float = float(np.random.beta(cfg.alpha, cfg.alpha))

        x_mixed = torch.zeros_like(x_orig)
        for i in range(cfg.k):
            aug_img = copy.deepcopy(img_pre)
            depth = random.randint(1, 3)
            for _ in range(depth):
                op = random.choice(_AUGMENTATIONS)
                aug_img = op(aug_img, cfg.severity)
            x_mixed = x_mixed + w[i] * self._to_tensor(aug_img)

        return m * x_orig + (1.0 - m) * x_mixed


# ---------------------------------------------------------------------------
# Worker init function for DataLoader multiprocessing
# ---------------------------------------------------------------------------

def augmix_worker_init_fn(worker_id: int) -> None:
    """Re-seed numpy's RNG per DataLoader worker to ensure augmentation diversity.

    Without this, all workers inherit the same numpy seed from the parent
    process and produce identical Dirichlet/Beta samples for the entire epoch.

    Pass this as ``worker_init_fn`` to the AugMix DataLoader.

    Args:
        worker_id: Index of the worker process (provided automatically by
                   PyTorch's DataLoader).
    """
    seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(seed % (2**32))
    random.seed(seed % (2**32))


# ---------------------------------------------------------------------------
# JSD consistency loss
# ---------------------------------------------------------------------------

class JSDLoss(nn.Module):
    """3-way Jensen-Shannon Divergence consistency loss for AugMix training.

    Computes the total loss as defined in Eq. 2 of Hendrycks et al.:

        L_total = CE(logits_orig, y) + lambda_jsd * JSD(p_orig, p_aug1, p_aug2)

    where the 3-way JSD is:

        JSD = (1/3) * [KL(p_orig||M) + KL(p_aug1||M) + KL(p_aug2||M)]
        M   = (p_orig + p_aug1 + p_aug2) / 3

    The CE term uses only the original (clean) logits.  The JSD term acts as
    a consistency regularizer encouraging the model to produce similar
    predictions for all three views.

    Args:
        num_classes: Number of output classes (unused internally but kept for
                     documentation / sanity checks).
        lambda_jsd: Weight for the JSD consistency term (paper appendix: 12.0).
        label_smoothing: Label smoothing epsilon passed to CrossEntropyLoss.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_jsd: float = 12.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_jsd = lambda_jsd
        self._ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self._kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        logits_orig: torch.Tensor,
        logits_aug1: torch.Tensor,
        logits_aug2: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AugMix total loss.

        Args:
            logits_orig: Raw model outputs for clean images, shape ``[B, C]``.
            logits_aug1: Raw model outputs for first AugMix view, shape ``[B, C]``.
            logits_aug2: Raw model outputs for second AugMix view, shape ``[B, C]``.
            targets: Ground-truth class indices, shape ``[B]``.

        Returns:
            Scalar loss tensor: CE(logits_orig, targets) + lambda_jsd * JSD.
        """
        # Cross-entropy on clean view only.
        ce_loss = self._ce(logits_orig, targets)

        # Softmax probabilities for mixture computation.
        p_orig = F.softmax(logits_orig, dim=1)
        p_aug1 = F.softmax(logits_aug1, dim=1)
        p_aug2 = F.softmax(logits_aug2, dim=1)

        # Mixture distribution M = (p_orig + p_aug1 + p_aug2) / 3.
        # M is a valid probability distribution (positive, sums to 1).
        M = (p_orig + p_aug1 + p_aug2) / 3.0

        # KLDivLoss expects log-probabilities as the first argument.
        jsd = (
            self._kl(F.log_softmax(logits_orig, dim=1), M)
            + self._kl(F.log_softmax(logits_aug1, dim=1), M)
            + self._kl(F.log_softmax(logits_aug2, dim=1), M)
        ) / 3.0

        return ce_loss + self.lambda_jsd * jsd
