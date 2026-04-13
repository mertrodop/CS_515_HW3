"""Test module for evaluating trained models on CIFAR-10 or MNIST datasets."""

import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from train import get_transforms


# ---------------------------------------------------------------------------
# CIFAR-10-C evaluation helpers
# ---------------------------------------------------------------------------

CIFAR10C_CORRUPTIONS: list[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


class _NumpyImageDataset(Dataset):
    """Lightweight Dataset wrapper around a numpy uint8 image array.

    Converts each image to a PIL Image on access and applies a transform
    before returning, avoiding loading all tensors into memory at once.

    Args:
        images: Numpy array of shape ``[N, H, W, C]`` with dtype uint8.
        labels: Numpy array of shape ``[N]`` with integer class indices.
        transform: Torchvision transform applied per sample.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose,
    ) -> None:
        """Store raw arrays and the transform to apply per sample.

        Args:
            images: Numpy array of shape ``[N, H, W, C]`` with dtype uint8.
            labels: Numpy array of shape ``[N]`` with integer class indices.
            transform: Torchvision transform applied to each PIL image.
        """
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return the transformed image and its label at *idx*."""
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])


@torch.no_grad()
def evaluate_cifar10c(
    model: nn.Module,
    params: dict,
    device: torch.device,
    cifar10c_path: str,
) -> dict[str, float]:
    """Evaluate a trained model on all CIFAR-10-C corruption types.

    Loads each of the 15 corruption .npy files (shape ``[50000, 32, 32, 3]``),
    splits them into 5 severity levels (10 000 images each), and computes the
    error rate at every (corruption, severity) pair.  Reports per-corruption
    mean error and the overall Mean Corruption Error (mCE).

    Only the standard normalization transform is applied at test time — no
    random augmentations.

    Args:
        model: Trained model already moved to ``device`` and in eval mode.
        params: Configuration dict; uses ``batch_size``, ``num_workers``,
                ``mean``, and ``std``.
        device: Computation device.
        cifar10c_path: Path to the CIFAR-10-C directory that contains one
                       ``.npy`` file per corruption type and ``labels.npy``.

    Returns:
        Dict mapping each corruption name to its mean error rate across all
        five severity levels.  Also prints a formatted results table.
    """
    labels_path = os.path.join(cifar10c_path, "labels.npy")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"labels.npy not found in CIFAR-10-C directory: {cifar10c_path}"
        )
    labels_all: np.ndarray = np.load(labels_path)  # (50000,)

    # Test-time transform: ToTensor + Normalize only (no spatial augmentation).
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params["mean"], params["std"]),
    ])

    results: dict[str, float] = {}
    col_w = max(len(c) for c in CIFAR10C_CORRUPTIONS)

    print("\n=== CIFAR-10-C Evaluation ===")
    print(f"  {'Corruption':<{col_w}}  {'Sev1':>6} {'Sev2':>6} {'Sev3':>6} "
          f"{'Sev4':>6} {'Sev5':>6}  {'mErr':>6}")
    print("  " + "-" * (col_w + 47))

    for corruption in CIFAR10C_CORRUPTIONS:
        data_path = os.path.join(cifar10c_path, f"{corruption}.npy")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Missing corruption file: {data_path}"
            )
        data: np.ndarray = np.load(data_path)  # (50000, 32, 32, 3), uint8

        severity_errors: list[float] = []
        for severity in range(1, 6):
            start = (severity - 1) * 10_000
            end   = severity * 10_000
            imgs_sev   = data[start:end]
            labels_sev = labels_all[start:end]

            ds     = _NumpyImageDataset(imgs_sev, labels_sev, tf)
            loader = DataLoader(
                ds,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=params["num_workers"],
            )

            correct, total = 0, 0
            for imgs_batch, labs_batch in loader:
                imgs_batch = imgs_batch.to(device)
                labs_batch = labs_batch.to(device)
                preds = model(imgs_batch).argmax(1)
                correct += preds.eq(labs_batch).sum().item()
                total   += imgs_batch.size(0)

            error = 1.0 - correct / total
            severity_errors.append(error)

        mean_error = float(np.mean(severity_errors))
        results[corruption] = mean_error

        sev_str = "  ".join(f"{e:.4f}" for e in severity_errors)
        print(f"  {corruption:<{col_w}}  {sev_str}  {mean_error:.4f}")

    mce = float(np.mean(list(results.values())))
    print("  " + "-" * (col_w + 47))
    print(f"  {'Mean Corruption Error (mCE)':<{col_w + 47 - 7}}{mce:.4f}\n")

    return results


@torch.no_grad()
def run_test(model: nn.Module, params: dict, device: torch.device) -> None:
    """Evaluate a trained model on the test split and print per-class accuracy.

    Loads model weights from ``params["save_path"]``, runs inference over the
    full test set, and reports overall accuracy together with per-class
    breakdown.

    Args:
        model: The neural network to evaluate.
        params: Configuration dictionary produced by ``get_params()``.  Must
            contain keys: ``dataset``, ``data_dir``, ``batch_size``,
            ``num_workers``, ``save_path``, and ``num_classes``.
        device: Device on which to run inference (CPU, CUDA, or MPS).
    """
    tf = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        test_ds = datasets.MNIST(params["data_dir"], train=False, download=True, transform=tf)
    else:  # cifar10
        test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=params["batch_size"],
                        shuffle=False, num_workers=params["num_workers"])

    model.load_state_dict(torch.load(params["save_path"], map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct = [0] * params["num_classes"]
    class_total   = [0] * params["num_classes"]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(params["num_classes"]):
        acc = class_correct[i] / class_total[i]
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    if params.get("cifar10c_path") is not None:
        evaluate_cifar10c(model, params, device, params["cifar10c_path"])
