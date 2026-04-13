"""t-SNE visualization of clean vs adversarial feature representations.

Extracts penultimate-layer features from a ResNet model for a subset of
clean and adversarial images, then produces two t-SNE plots:

  * ``tsne_{tag}_{norm}_by_class.png``  — coloured by true class (10 colours),
    marker shape distinguishes clean (circle) vs adversarial (cross).
  * ``tsne_{tag}_{norm}_clean_vs_adv.png`` — coloured by clean (blue) vs
    adversarial (red); marker shape encodes class is NOT shown here.

Feature extraction
------------------
A forward hook is registered on ``model.avgpool`` to capture the 512-dim
pooled feature vector before the final linear classifier.  Subsampling is
done *before* running t-SNE to keep runtime manageable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TSNEConfig:
    """Hyperparameters for the t-SNE visualisation.

    Attributes:
        n_samples: Number of *clean* images to use (adversarial count matches).
            Total points fed to t-SNE = ``2 * n_samples``.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of t-SNE optimisation iterations.
        random_state: Seed for reproducibility.
    """
    n_samples: int   = 1000
    perplexity: float = 30.0
    n_iter: int       = 1000
    random_state: int = 42


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    adv_images: Optional[torch.Tensor] = None,
    n_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract penultimate-layer (avgpool output) features.

    Registers a forward hook on ``model.avgpool`` to capture the 512-dim
    representation produced after global-average-pooling and before the
    final linear classifier.

    A random subset of ``n_samples`` images is drawn *before* t-SNE so
    that the structure is preserved (subsampling after would bias results).

    Args:
        model: Trained ResNet-18 model.
        dataloader: DataLoader yielding ``(normalized_images, labels)``.
        device: Computation device.
        adv_images: Pre-computed adversarial images as a flat tensor
            ``[N, C, H, W]`` (same order as the images produced by iterating
            ``dataloader``).  If ``None``, only clean features are extracted.
        n_samples: Number of clean (and adversarial) samples to keep.

    Returns:
        Tuple ``(features, labels, is_adv)`` where:
          * ``features`` — shape ``(M, 512)`` concatenated clean + adv.
          * ``labels``   — shape ``(M,)`` true class indices.
          * ``is_adv``   — shape ``(M,)`` binary flag (0 = clean, 1 = adv).
    """
    model.eval()

    # ── Register hook on avgpool ─────────────────────────────────────────────
    pooled_feats: list[torch.Tensor] = []

    def _hook(module, input, output):
        # output: (B, 512, 1, 1) → squeeze to (B, 512)
        pooled_feats.append(output.detach().squeeze(-1).squeeze(-1).cpu())

    handle = model.avgpool.register_forward_hook(_hook)

    # ── Collect clean features ───────────────────────────────────────────────
    all_feats: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            model(imgs)           # triggers hook
            all_feats.append(pooled_feats[-1])
            all_labels.append(lbls)

    handle.remove()
    pooled_feats.clear()

    clean_feats  = torch.cat(all_feats,  dim=0).numpy()   # (N, 512)
    clean_labels = torch.cat(all_labels, dim=0).numpy()   # (N,)

    N = len(clean_feats)
    n_use = min(n_samples, N)
    idx = np.random.default_rng(42).choice(N, n_use, replace=False)
    clean_feats  = clean_feats[idx]
    clean_labels = clean_labels[idx]

    if adv_images is None:
        is_adv   = np.zeros(n_use, dtype=np.int32)
        return clean_feats, clean_labels, is_adv

    # ── Collect adversarial features ────────────────────────────────────────
    # adv_images is the full adversarial set; select the same indices
    adv_subset = adv_images[idx].to(device)
    adv_feats_list: list[torch.Tensor] = []

    handle2 = model.avgpool.register_forward_hook(
        lambda m, i, o: adv_feats_list.append(o.detach().squeeze(-1).squeeze(-1).cpu())
    )

    bs = 256
    with torch.no_grad():
        for start in range(0, len(adv_subset), bs):
            model(adv_subset[start: start + bs])

    handle2.remove()

    adv_feats = torch.cat(adv_feats_list, dim=0).numpy()   # (n_use, 512)

    features = np.concatenate([clean_feats, adv_feats],   axis=0)  # (2*n_use, 512)
    labels   = np.concatenate([clean_labels, clean_labels], axis=0)  # same true labels
    is_adv   = np.concatenate([
        np.zeros(n_use, dtype=np.int32),
        np.ones(n_use,  dtype=np.int32),
    ], axis=0)

    return features, labels, is_adv


# ---------------------------------------------------------------------------
# t-SNE plotting
# ---------------------------------------------------------------------------

def run_tsne_and_plot(
    features: np.ndarray,
    labels: np.ndarray,
    is_adv: np.ndarray,
    output_dir: str = '.',
    tag: str = 'standard',
    norm: str = 'linf',
    cfg: TSNEConfig | None = None,
) -> None:
    """Fit t-SNE and produce two visualisation plots.

    The t-SNE embedding is fitted on the *combined* clean + adversarial
    feature matrix so both point sets share the same 2-D space.

    **Plot 1 — by class** (``tsne_{tag}_{norm}_by_class.png``):
      * Colour encodes true class (10 CIFAR-10 colours).
      * Filled circles (``o``) = clean; crosses (``X``) = adversarial.

    **Plot 2 — clean vs adv** (``tsne_{tag}_{norm}_clean_vs_adv.png``):
      * Blue = clean, red = adversarial.
      * All markers are the same shape.

    Args:
        features: Combined feature matrix, shape ``(M, D)``.
        labels: True class indices, shape ``(M,)``.
        is_adv: Binary flag, shape ``(M,)``; 0 = clean, 1 = adversarial.
        output_dir: Directory for output files.
        tag: Model tag used in filenames (e.g. ``'standard'``, ``'augmix'``).
        norm: Attack norm used in filenames (e.g. ``'linf'``, ``'l2'``).
        cfg: :class:`TSNEConfig`; uses defaults if ``None``.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if cfg is None:
        cfg = TSNEConfig()

    print(f'[t-SNE] Fitting on {len(features)} points (tag={tag}, norm={norm}) …')
    tsne = TSNE(
        n_components=2,
        perplexity=cfg.perplexity,
        max_iter=cfg.n_iter,
        random_state=cfg.random_state,
        init='pca',
        learning_rate='auto',
    )
    embedding = tsne.fit_transform(features)   # (M, 2)
    print('[t-SNE] Done.')

    clean_mask = is_adv == 0
    adv_mask   = is_adv == 1

    # ── Plot 1: coloured by class ────────────────────────────────────────────
    cmap10 = plt.cm.get_cmap('tab10', 10)
    colours = [cmap10(c) for c in labels]

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    # Clean: filled circles
    scatter_c = ax1.scatter(
        embedding[clean_mask, 0], embedding[clean_mask, 1],
        c=[cmap10(labels[i]) for i in np.where(clean_mask)[0]],
        marker='o', s=12, alpha=0.6, linewidths=0, label='_nolegend_',
    )
    # Adversarial: X markers
    scatter_a = ax1.scatter(
        embedding[adv_mask, 0], embedding[adv_mask, 1],
        c=[cmap10(labels[i]) for i in np.where(adv_mask)[0]],
        marker='X', s=14, alpha=0.6, linewidths=0, label='_nolegend_',
    )

    # Class legend patches
    import matplotlib.patches as mpatches
    class_handles = [
        mpatches.Patch(color=cmap10(k), label=CIFAR10_CLASSES[k])
        for k in range(10)
    ]
    # Shape legend
    import matplotlib.lines as mlines
    shape_handles = [
        mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                      markersize=7, label='Clean'),
        mlines.Line2D([], [], color='gray', marker='X', linestyle='None',
                      markersize=7, label='Adversarial'),
    ]
    leg1 = ax1.legend(handles=class_handles, title='Class', fontsize=7,
                      loc='upper left', bbox_to_anchor=(1.01, 1))
    ax1.add_artist(leg1)
    ax1.legend(handles=shape_handles, title='Type', fontsize=8, loc='upper left',
               bbox_to_anchor=(1.01, 0.35))

    ax1.set_title(f't-SNE — {tag} model, PGD20-{norm.upper()}\nColoured by class',
                  fontsize=12)
    ax1.set_xlabel('t-SNE dim 1')
    ax1.set_ylabel('t-SNE dim 2')
    ax1.grid(alpha=0.3)

    path1 = os.path.join(output_dir, f'tsne_{tag}_{norm}_by_class.png')
    plt.tight_layout()
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  Saved {path1}')

    # ── Plot 2: coloured by clean vs adversarial ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(
        embedding[clean_mask, 0], embedding[clean_mask, 1],
        c='steelblue', marker='o', s=12, alpha=0.5, label='Clean',
    )
    ax2.scatter(
        embedding[adv_mask, 0], embedding[adv_mask, 1],
        c='tomato', marker='o', s=12, alpha=0.5, label='Adversarial',
    )
    ax2.legend(fontsize=10)
    ax2.set_title(f't-SNE — {tag} model, PGD20-{norm.upper()}\nClean vs Adversarial',
                  fontsize=12)
    ax2.set_xlabel('t-SNE dim 1')
    ax2.set_ylabel('t-SNE dim 2')
    ax2.grid(alpha=0.3)

    path2 = os.path.join(output_dir, f'tsne_{tag}_{norm}_clean_vs_adv.png')
    plt.tight_layout()
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved {path2}')
