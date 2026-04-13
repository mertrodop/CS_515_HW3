"""PGD (Projected Gradient Descent) adversarial attack.

Implements Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks", ICLR 2018.

Normalization strategy
-----------------------
Model inputs are already normalized with CIFAR-10 mean/std.  The epsilon
budget is defined in pixel space [0, 1].  This module works by:
  1. Denormalizing inputs to pixel space [0, 1].
  2. Maintaining delta (the perturbation) in pixel space.
  3. On each step, renormalizing (x_nat + delta) before the forward pass.
  4. Autograd computes d(loss)/d(delta) through the normalization layer via
     the chain rule automatically — no manual scaling is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# CIFAR-10 normalization constants
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PGDConfig:
    """Configuration for a single PGD attack run.

    Attributes:
        norm: Threat model — ``'linf'`` or ``'l2'``.
        epsilon: Perturbation budget in pixel space [0, 1].
        num_steps: Number of PGD iterations.
        step_size: Per-step learning rate in pixel space.
            Recommended: ``2.5 * epsilon / num_steps``.
        random_start: Whether to initialise delta randomly inside the
            epsilon-ball before the first step.
    """
    norm: str
    epsilon: float
    num_steps: int
    step_size: float
    random_start: bool = True


# Preset configs matching the homework specification
PGD20_LINF = PGDConfig(
    norm='linf',
    epsilon=4 / 255,
    num_steps=20,
    step_size=2.5 * (4 / 255) / 20,
)

PGD20_L2 = PGDConfig(
    norm='l2',
    epsilon=0.25,
    num_steps=20,
    step_size=2.5 * 0.25 / 20,
)


# ---------------------------------------------------------------------------
# Core attack
# ---------------------------------------------------------------------------

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    cfg: PGDConfig,
    device: torch.device,
    mean: tuple[float, ...] = _CIFAR10_MEAN,
    std: tuple[float, ...]  = _CIFAR10_STD,
) -> torch.Tensor:
    """Run PGD on a batch of normalized images and return adversarial images.

    The perturbation delta is maintained in pixel space so that epsilon can
    be interpreted directly as a [0,1] pixel budget.  Autograd propagates
    gradients through the re-normalization on every step.

    Args:
        model: Target model in eval mode.  Must accept normalized inputs.
        images: Batch of normalized images, shape ``[B, C, H, W]``.
        labels: True class indices, shape ``[B]``.
        cfg: :class:`PGDConfig` specifying norm, epsilon, steps, step size.
        device: Computation device.
        mean: Per-channel normalization mean (default: CIFAR-10).
        std: Per-channel normalization std (default: CIFAR-10).

    Returns:
        Adversarial images in the *same normalized space* as the input,
        shape ``[B, C, H, W]``.
    """
    model.eval()

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1).to(device)

    # Denormalize to pixel space [0, 1]
    x_nat = (images.detach() * std_t + mean_t).clamp(0.0, 1.0)

    # ── Random initialisation ───────────────────────────────────────────────
    if cfg.random_start:
        if cfg.norm == 'linf':
            delta = torch.empty_like(x_nat).uniform_(-cfg.epsilon, cfg.epsilon)
        else:  # l2
            delta = torch.randn_like(x_nat)
            flat  = delta.view(delta.shape[0], -1)
            norms = flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-8)
            delta = delta / norms * cfg.epsilon
        # Keep delta inside the epsilon-ball AND the [0,1] image box
        delta = ((x_nat + delta).clamp(0.0, 1.0) - x_nat).detach()
    else:
        delta = torch.zeros_like(x_nat)

    # ── PGD iterations ──────────────────────────────────────────────────────
    for _ in range(cfg.num_steps):
        delta.requires_grad_(True)

        # Renormalize for model forward pass
        x_adv_norm = (x_nat + delta - mean_t) / std_t
        logits = model(x_adv_norm)
        loss   = F.cross_entropy(logits, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad = delta.grad.detach()

            if cfg.norm == 'linf':
                # Signed gradient step + L-inf projection
                delta = delta + cfg.step_size * grad.sign()
                delta = delta.clamp(-cfg.epsilon, cfg.epsilon)
            else:  # l2
                # Normalised gradient step + L2 ball projection
                flat      = grad.view(grad.shape[0], -1)
                grad_norm = flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-8)
                delta     = delta + cfg.step_size * grad / grad_norm

                flat      = delta.view(delta.shape[0], -1)
                delta_norm = flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-8)
                # Project onto L2 ball of radius epsilon
                delta = delta * cfg.epsilon / delta_norm.clamp(min=cfg.epsilon)

            # Project onto image box [0, 1] in pixel space
            delta = ((x_nat + delta).clamp(0.0, 1.0) - x_nat).detach()

    # Return adversarial images renormalized to model input space
    with torch.no_grad():
        x_adv_norm = (x_nat + delta - mean_t) / std_t
    return x_adv_norm.detach()


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_adversarial(
    model: nn.Module,
    loader: DataLoader,
    cfg: PGDConfig,
    device: torch.device,
    max_samples: int | None = None,
    mean: tuple[float, ...] = _CIFAR10_MEAN,
    std: tuple[float, ...]  = _CIFAR10_STD,
) -> dict:
    """Evaluate adversarial robustness over a DataLoader.

    For each batch, runs :func:`pgd_attack` and records clean and adversarial
    accuracy.  Also collects samples where the clean prediction was correct
    but the adversarial prediction was wrong (misclassified due to the
    perturbation) — useful for Grad-CAM analysis.

    Args:
        model: Target model.
        loader: DataLoader yielding ``(images, labels)`` batches of normalized
            CIFAR-10 images.
        cfg: :class:`PGDConfig` for the attack.
        device: Computation device.
        max_samples: Stop after this many images (``None`` = full loader).
        mean: CIFAR-10 per-channel mean.
        std: CIFAR-10 per-channel std.

    Returns:
        Dict with keys:
            ``'clean_acc'``, ``'adv_acc'``, ``'num_fooled'``,
            ``'misclassified'`` — list of
            ``(clean_img, adv_img, true_lbl, clean_pred, adv_pred)`` tuples
            (all CPU tensors, single images not batches).
    """
    model.eval()

    clean_correct = adv_correct = n = 0
    misclassified: list = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_preds = model(images).argmax(1)

        # Adversarial accuracy
        adv_images = pgd_attack(model, images, labels, cfg, device, mean, std)
        with torch.no_grad():
            adv_preds = model(adv_images).argmax(1)

        clean_correct += clean_preds.eq(labels).sum().item()
        adv_correct   += adv_preds.eq(labels).sum().item()
        n             += images.size(0)

        # Collect samples: clean correct AND adv wrong
        mask = clean_preds.eq(labels) & adv_preds.ne(labels)
        for idx in mask.nonzero(as_tuple=False).squeeze(1):
            i = idx.item()
            misclassified.append((
                images[i].cpu(),
                adv_images[i].cpu(),
                labels[i].item(),
                clean_preds[i].item(),
                adv_preds[i].item(),
            ))

        print(f'  Batch {batch_idx+1:3d}/{len(loader)}  '
              f'clean: {clean_correct/n:.4f}  adv: {adv_correct/n:.4f}  '
              f'fooled so far: {clean_correct - adv_correct}')

        if max_samples is not None and n >= max_samples:
            break

    return {
        'clean_acc':     clean_correct / n,
        'adv_acc':       adv_correct   / n,
        'num_fooled':    clean_correct - adv_correct,
        'misclassified': misclassified,
    }
