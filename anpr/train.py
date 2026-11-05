"""Train the ResNet baseline or RGB-plus-soft-mask model on isolated plate crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from anpr.data import PlateDataset, load_manifest, split_records
from anpr.model import PlateResNet, save_checkpoint
from anpr.paths import protect_output_paths


def run_training(arguments: argparse.Namespace) -> dict:
    """Train against train/val only and save the checkpoint with lowest val loss.

    The optional segmentation loss requires paired masks. Test rows are never
    loaded by this function. A saved checkpoint records the manifest fingerprint
    and seed, but cannot prove quality on an independent camera domain.
    """
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.threads < 1:
        # Reject no-op or invalid runs before writing any training artifacts.
        raise ValueError("epochs, batch-size, and threads must be positive integers")
    if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    # Check checkpoint, temporary save and history destinations before reading or allocating a model.
    checkpoint_path = Path(arguments.output)
    output_paths = [
        checkpoint_path,
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp"),
        checkpoint_path.with_suffix(".training.json"),
    ]
    protect_output_paths([arguments.manifest], output_paths)
    # Reproduce initialization, shuffle order, and augmentation on a fixed runtime.
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    # Deterministic kernels avoid silent replay differences on supported backends.
    torch.use_deterministic_algorithms(True)
    # Split source/identity components before any DataLoader sees a crop.
    records = load_manifest(arguments.manifest, max_length=arguments.max_length)
    # A filename typo must not replace a crop or segmentation target during the first checkpoint save.
    input_paths = [arguments.manifest] + [record.path for record in records]
    input_paths.extend(record.mask_path for record in records if record.mask_path is not None)
    protect_output_paths(input_paths, output_paths)
    partitions = split_records(records, arguments.seed)
    if not partitions["train"] or not partitions["val"]:
        # A meaningful validation checkpoint needs at least two independent components.
        raise ValueError("Training requires nonempty, isolated train and val splits")
    if arguments.use_mask and any(
        record.mask_path is None for name in ("train", "val") for record in partitions[name]
    ):
        # Placeholder masks exist only for batching and must never supervise this model.
        raise ValueError("--use-mask requires masks for every training and validation crop")
    # Private loader RNG makes shuffle ordering independent of model initialization.
    loader_generator = torch.Generator().manual_seed(arguments.seed)
    # Only training data receives random corruption; validation remains a fixed reference.
    train_loader = DataLoader(
        PlateDataset(partitions["train"], arguments.max_length, augment=True, seed=arguments.seed),
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        PlateDataset(partitions["val"], arguments.max_length),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )
    # The device is explicit so CPU smoke runs never imply available GPU training.
    device = torch.device(arguments.device)
    model = PlateResNet(
        max_length=arguments.max_length,
        pretrained=arguments.pretrained,
        use_mask=arguments.use_mask,
        head_layout=getattr(arguments, "head_layout", "global"),
    ).to(device)
    # AdamW is a conventional baseline; tuning is intentionally separate from test evaluation.
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    # Manifest bytes identify labels and split assignments used by this run.
    manifest_sha256 = hashlib.sha256(Path(arguments.manifest).read_bytes()).hexdigest()
    # Validation loss chooses a checkpoint without inspecting the held-out test set.
    best_val_loss = math.inf
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(arguments.epochs):
        # Training loss combines recognition with optional clean-mask supervision.
        train_loader.dataset.set_epoch(epoch)
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in train_loader:
            # Batch images/targets are the only recognizer training inputs.
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            # Transpose logits to the class-first convention expected by CE.
            logits = model(images)
            loss = functional.cross_entropy(logits.transpose(1, 2), targets)
            if arguments.use_mask:
                # Keep original RGB evidence while teaching a soft glyph foreground channel.
                masks = batch["mask"].to(device)
                loss = loss + 0.2 * functional.binary_cross_entropy_with_logits(
                    model.mask_logits(images), masks
                )
            if not torch.isfinite(loss):
                # Stop corrupt training before saving non-finite model parameters.
                raise RuntimeError("Training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * images.shape[0]
            train_count += images.shape[0]
        # Evaluate the same objective without gradients or stochastic training transforms.
        model.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                # Validation crops remain isolated from optimizer updates.
                images = batch["image"].to(device)
                targets = batch["targets"].to(device)
                logits = model(images)
                loss = functional.cross_entropy(logits.transpose(1, 2), targets)
                if arguments.use_mask:
                    # Match the auxiliary objective used for the optional mask model.
                    masks = batch["mask"].to(device)
                    loss = loss + 0.2 * functional.binary_cross_entropy_with_logits(
                        model.mask_logits(images), masks
                    )
                val_loss_total += loss.item() * images.shape[0]
                val_count += images.shape[0]
        # History exposes the actual sample-weighted losses for each completed epoch.
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss_total / train_count,
            "val_loss": val_loss_total / val_count,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if not math.isfinite(epoch_metrics["val_loss"]):
            raise RuntimeError("Validation produced a non-finite loss")
        if epoch_metrics["val_loss"] < best_val_loss:
            # Persist only a reviewable best-validation checkpoint, never the test-selected epoch.
            best_val_loss = epoch_metrics["val_loss"]
            best_epoch = epoch + 1
            save_checkpoint(
                model,
                arguments.output,
                temperature=1.0,
                metadata={
                    "seed": arguments.seed,
                    "epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "manifest_sha256": manifest_sha256,
                    "manifest": str(Path(arguments.manifest).resolve()),
                    "split_counts": {name: len(partition) for name, partition in partitions.items()},
                    # Retain split membership so a different evaluation seed cannot reuse training identities.
                    "split_provenance": {
                        name: [
                            {"plate": record.plate, "group": record.group, "path": str(record.path)}
                            for record in partition
                        ]
                        for name, partition in partitions.items()
                    },
                    "augmentation": "paired crop rotation/translation plus brightness, contrast, defocus, motion, downsample, JPEG",
                    "pretrained": arguments.pretrained,
                    "mask_loss_weight": 0.2 if arguments.use_mask else 0.0,
                    "calibrated": False,
                    # Plain strings avoid pickling TorchVersion objects in weights-only checkpoints.
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "numpy": str(np.__version__),
                    },
                },
            )
    # Persist the complete learning history instead of requiring terminal output to reconstruct a run.
    training_report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint": str(arguments.output),
        "manifest_sha256": manifest_sha256,
        "seed": arguments.seed,
        "split_counts": {name: len(partition) for name, partition in partitions.items()},
        "use_mask": arguments.use_mask,
        "head_layout": getattr(arguments, "head_layout", "global"),
        "pretrained": arguments.pretrained,
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "epochs": arguments.epochs,
    }
    # Keep training diagnostics beside the checkpoint selected by validation loss.
    report_path = Path(arguments.output).with_suffix(".training.json")
    report_path.write_text(json.dumps(training_report, indent=2, allow_nan=False) + "\n")
    return training_report


def main() -> None:
    """Expose a reproducible local baseline training command."""
    # CLI arguments state architecture and data assumptions explicitly.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", help="Download ImageNet ResNet18 initialization")
    parser.add_argument(
        "--head-layout",
        choices=("global", "spatial"),
        default="global",
        help="Keep the original globally pooled heads or preserve left-to-right features",
    )
    parser.add_argument(
        "--use-mask", action="store_true", help="Train RGB plus a U-Net soft mask with paired targets"
    )
    # Parsed options are passed as one immutable-in-practice run configuration.
    arguments = parser.parse_args()
    try:
        # Value errors become actionable CLI usage failures instead of partial runs.
        print(json.dumps(run_training(arguments)), flush=True)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Training failed: {error}\n")


if __name__ == "__main__":
    # Importing train must not start downloads, allocate a model, or run an epoch.
    main()
