import argparse


def get_params() -> dict:
    """Parse CLI arguments and return a unified params dict for training/testing."""
    parser = argparse.ArgumentParser(description="Deep Learning on MNIST / CIFAR-10")

    parser.add_argument("--mode",      choices=["train", "test", "both"], default="both")
    parser.add_argument("--dataset",   choices=["mnist", "cifar10"],      default="mnist")
    parser.add_argument("--model", choices=["mlp", "cnn", "vgg", "resnet", "mobilenet", "transfer"], default="mlp")
    # Transfer learning-specific
    parser.add_argument("--transfer_backbone", choices=["resnet18", "vgg16"], default="resnet18",
                        help="Pretrained backbone to use when --model=transfer")
    parser.add_argument("--pretrained_option", type=int, choices=[1, 2], default=1,
                        help="1 = resize to 224 + freeze backbone, 2 = modify early convs + train all")
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--batch_size",type=int,   default=64)
    # VGG-specific
    parser.add_argument("--vgg_depth", choices=["11", "13", "16", "19"], default="16")
    # ResNet-specific: map a simple int to a block config
    parser.add_argument("--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
                        metavar=("L1", "L2", "L3", "L4"),
                        help="Number of blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.0 = standard CE, 0.1 = Szegedy et al.)")
    # AugMix / CIFAR-10-C arguments
    parser.add_argument("--use-augmix", action="store_true", default=False,
                        help="Enable AugMix data augmentation during CIFAR-10 training.")
    parser.add_argument("--augmix-alpha", type=float, default=1.0,
                        help="Dirichlet/Beta concentration parameter for AugMix (default: 1.0).")
    parser.add_argument("--augmix-severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Augmentation operation severity for AugMix (default: 3).")
    parser.add_argument("--save-path", type=str, default="best_model.pth",
                        help="Path to save/load the best model checkpoint (default: best_model.pth).")
    parser.add_argument("--cifar10c-path", type=str, default=None,
                        help="Path to the CIFAR-10-C directory containing .npy corruption files.")

    args = parser.parse_args()

    # Dataset-dependent settings
    if args.dataset == "mnist":
        input_size = 784          # 1 × 28 × 28
        mean, std  = (0.1307,), (0.3081,)
    else:                         # cifar10
        input_size = 3072         # 3 × 32 × 32
        mean       = (0.4914, 0.4822, 0.4465)
        std        = (0.2023, 0.1994, 0.2010)

    return {
        # Data
        "dataset":      args.dataset,
        "data_dir":     "./data",
        "num_workers":  2,
        "mean":         mean,
        "std":          std,

        # Model
        "model":        args.model,
        "input_size":   input_size,
        "hidden_sizes": [512, 256, 128],
        "num_classes":  10,
        "dropout":      0.3,
        "vgg_depth":    args.vgg_depth,
        "resnet_layers": args.resnet_layers,
        "transfer_backbone": args.transfer_backbone,
        "pretrained_option": args.pretrained_option,
        "resize_224":        False,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":         42,
        "device":       args.device,
        "save_path":    args.save_path,
        "log_interval": 100,

        # CLI
        "mode":         args.mode,

        # Label smoothing
        "label_smoothing": args.label_smoothing,

        # AugMix
        "use_augmix":      args.use_augmix,
        "augmix_alpha":    args.augmix_alpha,
        "augmix_severity": args.augmix_severity,

        # CIFAR-10-C evaluation
        "cifar10c_path":   args.cifar10c_path,
    }