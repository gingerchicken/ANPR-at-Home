"""Train the recurrent ResNet CTC experiment without fixed-position padding targets."""

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
from torch import nn
from torch.utils.data import DataLoader

from anpr.alphabet import NUM_CLASSES, encode_plate, normalize_plate
from anpr.ctc_model import PlateCTC
from anpr.data import PlateDataset, load_manifest, split_records
from anpr.model import save_checkpoint
from anpr.paths import protect_output_paths


def minimum_ctc_frames(plate: str) -> int:
    """Return the shortest CTC path that can emit this complete registration.

    Adjacent equal symbols require a separating blank. A repeated letter is not
    removed from the target; it adds one required alignment frame instead.
    """
    # Display separators are normalized in the same way as all other training labels.
    normalized_plate = normalize_plate(plate)
    return len(normalized_plate) + sum(
        first == second for first, second in zip(normalized_plate, normalized_plate[1:])
    )


def prepare_ctc_targets(
    padded_targets: torch.Tensor, plates: list[str], time_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate padded dataset labels, then return flat nonblank targets and CPU lengths.

    The shared dataset still pads batches for the positional baseline. This
    boundary validates that padding against the true text before removing only
    the trailing zero entries. Impossible alignments raise ValueError.
    """
    if padded_targets.ndim != 2 or padded_targets.shape[0] != len(plates) or not plates:
        # Silent batch/label misalignment would teach unrelated registrations to image features.
        raise ValueError("CTC targets require a nonempty two-dimensional batch matching plate labels")
    if padded_targets.dtype != torch.long:
        # Floating labels must not be rounded silently into a different target alphabet.
        raise ValueError("CTC targets must use integer torch.long class indices")
    if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps < 1:
        # Positive frame capacity is a required alignment invariant, not a tunable loss workaround.
        raise ValueError("CTC alignment needs a positive integer number of frames")
    # The caller's padded tensor remains untouched; each slice below retains repeated target symbols.
    flat_parts = []
    target_lengths = []
    for row_index, supplied_plate in enumerate(plates):
        # The source label defines exactly how much trailing padding may be removed.
        plate = normalize_plate(supplied_plate)
        expected_indices = torch.tensor(
            encode_plate(plate, padded_targets.shape[1]), dtype=torch.long, device=padded_targets.device
        )
        if not torch.equal(padded_targets[row_index], expected_indices):
            raise ValueError("Padded target indices do not match the supplied plate label")
        if minimum_ctc_frames(plate) > time_steps:
            # zero_infinity would hide this defect by discarding supervision for the impossible sample.
            raise ValueError(f"Plate {plate!r} requires more than {time_steps} CTC alignment frames")
        # Real class indices are strictly positive; class zero appears only in model frame outputs.
        flat_parts.append(padded_targets[row_index, : len(plate)])
        target_lengths.append(len(plate))
    return torch.cat(flat_parts), torch.tensor(target_lengths, dtype=torch.long)


def ctc_batch_loss(logits: torch.Tensor, padded_targets: torch.Tensor, plates: list[str]) -> torch.Tensor:
    """Compute mean target-length-normalized CTC loss while retaining impossible-label errors."""
    if logits.ndim != 3 or logits.shape[0] != len(plates) or logits.shape[2] != NUM_CLASSES:
        # A positional model output must not accidentally be interpreted with the CTC frame contract.
        raise ValueError("Expected CTC logits shaped batch, frames, 37 classes")
    # Explicit target validation prevents internal blanks or target truncation before CTCLoss runs.
    flat_targets, target_lengths = prepare_ctc_targets(padded_targets, plates, logits.shape[1])
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    # PyTorch expects time-major log probabilities; class zero remains the genuine CTC blank.
    log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=False)
    return criterion(log_probabilities, flat_targets, input_lengths, target_lengths)


def run_training(arguments: argparse.Namespace) -> dict:
    """Train train/val splits only and persist the lowest-validation-loss CTC checkpoint.

    Test rows are recorded for isolation auditing but never loaded by an epoch.
    Max length and adjacent repeats are checked against the encoder's 24 frames
    before the first optimizer step; nonfinite loss is never silently zeroed.
    """
    # TODO(refactor): Share split validation, path protection and provenance reporting with the
    # positional trainer. Duplicated orchestration can drift between experiments; keep the two
    # loss paths explicit during this alignment comparison before changing the retained baseline.
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.threads < 1:
        # Reject invalid or empty runs before any output path can be written.
        raise ValueError("epochs, batch-size, and threads must be positive integers")
    if not math.isfinite(arguments.learning_rate) or arguments.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    # Check all output roles, including atomic save staging, against declared source ownership.
    checkpoint_path = Path(arguments.output)
    output_paths = [
        checkpoint_path,
        checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp"),
        checkpoint_path.with_suffix(".training.json"),
    ]
    protect_output_paths([arguments.manifest], output_paths)
    # Reproduce initialization and augmentations using the same seed/thread policy as the baseline.
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    torch.use_deterministic_algorithms(True)
    # Identity/source components stay together before any data reaches an optimizer.
    records = load_manifest(arguments.manifest, max_length=arguments.max_length)
    input_paths = [arguments.manifest] + [record.path for record in records]
    input_paths.extend(record.mask_path for record in records if record.mask_path is not None)
    protect_output_paths(input_paths, output_paths)
    partitions = split_records(records, arguments.seed)
    if not partitions["train"] or not partitions["val"]:
        # Checkpoint selection needs independent validation evidence, not training loss alone.
        raise ValueError("CTC training requires nonempty, isolated train and val splits")
    # The sequence model shares preprocessing and backbone kernels with the measured spatial baseline.
    device = torch.device(arguments.device)
    model = PlateCTC(max_length=arguments.max_length, pretrained=arguments.pretrained).to(device)
    for split_name in ("train", "val"):
        # Check all epoch participants before an invalid late batch can interrupt an otherwise partial run.
        for record in partitions[split_name]:
            if minimum_ctc_frames(record.plate) > model.time_steps:
                raise ValueError(f"Plate {record.plate!r} requires more than {model.time_steps} CTC frames")
    # A private shuffle RNG keeps batch ordering independent of recurrent-layer initialization.
    loader_generator = torch.Generator().manual_seed(arguments.seed)
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
    # Match the baseline optimizer and learning rate rather than tuning against the public test source.
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    manifest_sha256 = hashlib.sha256(Path(arguments.manifest).read_bytes()).hexdigest()
    # Selection uses the same target-length-normalized CTC objective as optimization.
    best_val_loss = math.inf
    best_epoch = 0
    history = []
    for epoch in range(arguments.epochs):
        # Epoch-specific paired augmentation remains deterministic without touching validation pixels.
        train_loader.dataset.set_epoch(epoch)
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in train_loader:
            # Targets use their original text for lengths, preserving repeated characters exactly.
            images = batch["image"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = ctc_batch_loss(logits, targets, batch["plate"])
            if not torch.isfinite(loss):
                raise RuntimeError("CTC training produced a non-finite loss")
            loss.backward()
            optimizer.step()
            # Weight batch means by sample count so the final short batch does not skew epoch means.
            train_loss_total += loss.item() * images.shape[0]
            train_count += images.shape[0]
        # Validation has no stochastic corruption or model state updates.
        model.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                # Test rows are absent from this loader and cannot affect checkpoint selection.
                images = batch["image"].to(device)
                targets = batch["targets"].to(device)
                logits = model(images)
                loss = ctc_batch_loss(logits, targets, batch["plate"])
                if not torch.isfinite(loss):
                    raise RuntimeError("CTC validation produced a non-finite loss")
                val_loss_total += loss.item() * images.shape[0]
                val_count += images.shape[0]
        # Epoch history names the loss scaling rather than comparing its magnitude directly with CE.
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss_total / train_count,
            "val_loss": val_loss_total / val_count,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if epoch_metrics["val_loss"] < best_val_loss:
            # Never use test recall or public benchmark predictions to select an epoch.
            best_val_loss = epoch_metrics["val_loss"]
            best_epoch = epoch + 1
            save_checkpoint(
                model,
                checkpoint_path,
                temperature=1.0,
                metadata={
                    "seed": arguments.seed,
                    "epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "manifest_sha256": manifest_sha256,
                    "manifest": str(Path(arguments.manifest).resolve()),
                    "split_counts": {name: len(partition) for name, partition in partitions.items()},
                    "split_provenance": {
                        name: [
                            {"plate": record.plate, "group": record.group, "path": str(record.path)}
                            for record in partition
                        ]
                        for name, partition in partitions.items()
                    },
                    "pretrained": arguments.pretrained,
                    "calibrated": False,
                    "objective": "CTC mean NLL divided by target length per sample; zero_infinity=False",
                    "augmentation": "training-only paired geometry plus defocus, motion, resolution and JPEG",
                    "runtime": {
                        "python": platform.python_version(),
                        "torch": str(torch.__version__),
                        "numpy": str(np.__version__),
                    },
                },
            )
    # Persist actual configuration and loss history beside the selected checkpoint.
    training_report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "manifest_sha256": manifest_sha256,
        "seed": arguments.seed,
        "split_counts": {name: len(partition) for name, partition in partitions.items()},
        "model_kind": "ctc",
        "head_layout": "ctc",
        "use_mask": False,
        "pretrained": arguments.pretrained,
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "epochs": arguments.epochs,
        "objective": "CTC mean NLL divided by target length per sample",
    }
    checkpoint_path.with_suffix(".training.json").write_text(
        json.dumps(training_report, indent=2, allow_nan=False) + "\n"
    )
    return training_report


def main() -> None:
    """Expose the reproducible CTC comparison command without starting work on import."""
    # CLI defaults match the measured broad-label spatial comparison where the objective allows it.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true", help="Download ImageNet ResNet18 initialization")
    # Parsed configuration owns no model or data resources until run_training is called explicitly.
    arguments = parser.parse_args()
    try:
        # Clear CLI failures retain their real cause instead of silently skipping impossible labels.
        print(json.dumps(run_training(arguments)), flush=True)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"CTC training failed: {error}\n")


if __name__ == "__main__":
    # Importing training helpers must not allocate a model, start an epoch or download weights.
    main()
