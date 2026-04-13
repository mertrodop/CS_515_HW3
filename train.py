import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(params: dict, train: bool = True) -> transforms.Compose:
    """Return a torchvision transform pipeline for the given dataset and split."""
    mean, std = params["mean"], params["std"]

    if params["dataset"] == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if params.get("resize_224", False):
            if train:
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2023, 0.1994, 0.2010),
                    ),
                ])
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(params: dict) -> tuple[DataLoader, DataLoader]:
    """Build and return (train_loader, val_loader) for the dataset specified in params."""
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    if params["dataset"] == "mnist":
        train_ds = datasets.MNIST(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(params["data_dir"], train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True,  num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    """Run one training epoch; return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on loader; return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
    return total_loss / n, correct / n


def augmix_collate_fn(
    batch: list,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Collate a batch of AugMix samples into stacked tensors.

    Each element of ``batch`` is ``((x_orig, x_aug1, x_aug2), label)`` where
    every image view is already a normalized tensor of shape ``[3, 32, 32]``.
    The default PyTorch collate cannot handle this nested tuple structure, so
    this custom function stacks each view independently.

    This function is defined at module level to be picklable, which is required
    when using ``num_workers > 0`` in a DataLoader.

    Args:
        batch: List of ``((x_orig, x_aug1, x_aug2), label)`` tuples.

    Returns:
        ``((x_orig_batch, x_aug1_batch, x_aug2_batch), labels)`` where each
        image batch has shape ``[N, 3, 32, 32]`` and labels has shape ``[N]``.
    """
    imgs_orig = torch.stack([item[0][0] for item in batch])
    imgs_aug1 = torch.stack([item[0][1] for item in batch])
    imgs_aug2 = torch.stack([item[0][2] for item in batch])
    labels    = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return (imgs_orig, imgs_aug1, imgs_aug2), labels


def get_augmix_loaders(params: dict) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders with AugMix augmentation on the training set.

    The training loader uses :class:`augmix.AugMixTransform` and a custom
    collate function that handles the per-sample 3-tuple output.  The
    validation loader uses the standard non-augmented transform and the
    default collate.

    Only CIFAR-10 is supported; call this only when ``params["use_augmix"]``
    is True and ``params["dataset"] == "cifar10"``.

    Args:
        params: Configuration dict from ``get_params()``.

    Returns:
        ``(train_loader, val_loader)`` where the train loader yields
        ``((x_orig, x_aug1, x_aug2), labels)`` per batch.
    """
    from augmix import AugMixConfig, AugMixTransform, augmix_worker_init_fn

    config = AugMixConfig(
        alpha=params.get("augmix_alpha", 1.0),
        severity=params.get("augmix_severity", 3),
    )
    augmix_tf = AugMixTransform(
        mean=params["mean"],
        std=params["std"],
        config=config,
    )
    val_tf = get_transforms(params, train=False)

    train_ds = datasets.CIFAR10(
        params["data_dir"], train=True, download=True, transform=augmix_tf
    )
    val_ds = datasets.CIFAR10(
        params["data_dir"], train=False, download=True, transform=val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        collate_fn=augmix_collate_fn,
        worker_init_fn=augmix_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )
    return train_loader, val_loader


def train_one_epoch_augmix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    """Run one AugMix training epoch using the JSD consistency loss.

    Expects the DataLoader to yield ``((x_orig, x_aug1, x_aug2), labels)``
    per batch (as produced by :func:`get_augmix_loaders`).

    All three views are concatenated along the batch dimension for a **single**
    forward pass, then the logits are split back.  This means the effective
    forward-pass batch size is ``3 * batch_size``; reduce ``--batch_size``
    if GPU memory is limited (e.g. ``--batch_size 21`` → 63 images per pass).

    Accuracy is computed on the clean (``x_orig``) logits only, matching
    standard evaluation semantics.

    Args:
        model: The neural network (in train mode after this call sets it).
        loader: AugMix DataLoader yielding 3-tuple image batches.
        optimizer: Optimizer for the model parameters.
        criterion: :class:`augmix.JSDLoss` instance.
        device: Computation device.
        log_interval: Number of batches between progress log lines.

    Returns:
        ``(avg_loss, accuracy)`` over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, ((x_orig, x_aug1, x_aug2), labels) in enumerate(loader):
        x_orig  = x_orig.to(device)
        x_aug1  = x_aug1.to(device)
        x_aug2  = x_aug2.to(device)
        labels  = labels.to(device)

        # Single forward pass over all three views concatenated on the batch dim.
        x_all      = torch.cat([x_orig, x_aug1, x_aug2], dim=0)  # (3B, C, H, W)
        logits_all = model(x_all)                                  # (3B, num_classes)
        B = x_orig.size(0)
        logits_orig = logits_all[:B]
        logits_aug1 = logits_all[B:2 * B]
        logits_aug2 = logits_all[2 * B:]

        loss = criterion(logits_orig, logits_aug1, logits_aug2, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * B
        correct    += logits_orig.argmax(1).eq(labels).sum().item()
        n          += B

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def run_training(model: nn.Module, params: dict, device: torch.device) -> dict:
    """Train model using params; save best checkpoint; return history dict."""
    use_augmix = params.get("use_augmix", False)
    if use_augmix:
        if params["dataset"] != "cifar10":
            raise ValueError("--use-augmix is only supported for the cifar10 dataset.")
        if params.get("resize_224", False):
            raise ValueError("--use-augmix is incompatible with --pretrained_option 1 "
                             "(resize_224=True). Use --pretrained_option 2 instead.")
        from augmix import JSDLoss
        train_loader, val_loader = get_augmix_loaders(params)
        criterion = JSDLoss(
            num_classes=params["num_classes"],
            lambda_jsd=12.0,
            label_smoothing=params.get("label_smoothing", 0.0),
        )
    else:
        train_loader, val_loader = get_loaders(params)
        criterion = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable,lr=params["learning_rate"],weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        if use_augmix:
            tr_loss, tr_acc = train_one_epoch_augmix(
                model, train_loader, optimizer, criterion, device, params["log_interval"]
            )
        else:
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, params["log_interval"]
            )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    return history


class DistillationLoss(nn.Module):
    """Hinton knowledge distillation loss.

    loss = alpha * T² * KL(student_soft || teacher_soft)
         + (1 - alpha) * CrossEntropy(student, hard_labels)
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7) -> None:
        """Initialize DistillationLoss.

        Args:
            temperature: Softening temperature T applied to both student and
                teacher logits before computing the KL divergence.
            alpha: Weight for the soft (distillation) loss.  The hard
                cross-entropy loss receives weight ``1 - alpha``.
        """
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the combined soft + hard distillation loss.

        Args:
            student_logits: Raw logits from the student model, shape ``(B, C)``.
            teacher_logits: Raw logits from the frozen teacher, shape ``(B, C)``.
            targets: Ground-truth class indices, shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        soft_loss = self.kl(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits  / self.T, dim=1),
        ) * (self.T ** 2)
        hard_loss = self.ce(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def run_kd_training(
    student: nn.Module,
    teacher: nn.Module,
    params: dict,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict:
    """Train *student* using knowledge distillation from a frozen *teacher*."""
    train_loader, val_loader = get_loaders(params)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if criterion is None:
        criterion = DistillationLoss(
            temperature=params.get("kd_temperature", 4.0),
            alpha=params.get("kd_alpha", 0.7),
        )
    ce_eval = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc     = 0.0
    best_weights = None
    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, params["epochs"] + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            optimizer.zero_grad()
            student_logits = student(imgs)
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += student_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % params["log_interval"] == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] "
                      f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        tr_loss, tr_acc = total_loss / n, correct / n
        val_loss, val_acc = validate(student, val_loader, ce_eval, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch}/{params['epochs']}  "
              f"train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    student.load_state_dict(best_weights)
    print(f"\nKD training done. Best val accuracy: {best_acc:.4f}")
    return history


class TeacherGuidedLSLoss(nn.Module):
    """Dynamic label smoothing KD using only teacher's true-class probability.

    Effective smoothing epsilon is example-wise:
      epsilon_i = 1 - softmax(teacher_logits/T)[i, y_i]

    Soft label for sample i:
      - true class:    p = softmax(teacher_logits / T)[i, y_i]
      - other classes: (1 - p) / (C - 1)  each (uniform)
    Loss = cross-entropy(student_logits, soft_labels)

    temperature > 1 → teacher more uniform → more smoothing overall.
    """
    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize TeacherGuidedLSLoss.

        Args:
            temperature: Temperature applied to teacher logits before computing
                softmax.  Values > 1 produce more uniform soft labels.
        """
        super().__init__()
        self.T = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss against teacher-derived soft labels.

        Args:
            student_logits: Raw logits from the student model, shape ``(B, C)``.
            teacher_logits: Raw logits from the frozen teacher, shape ``(B, C)``.
            targets: Ground-truth class indices, shape ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        B, C = student_logits.shape
        teacher_probs = nn.functional.softmax(teacher_logits / self.T, dim=1)
        p_true = teacher_probs[torch.arange(B, device=student_logits.device), targets]  # (B,)
        # build soft labels
        soft_labels = (1 - p_true).unsqueeze(1).expand(B, C) / (C - 1)
        soft_labels = soft_labels.clone()
        soft_labels[torch.arange(B, device=student_logits.device), targets] = p_true
        # cross-entropy with soft labels
        log_probs = nn.functional.log_softmax(student_logits, dim=1)
        return -(soft_labels * log_probs).sum(dim=1).mean()