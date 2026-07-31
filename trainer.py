#!/usr/bin/env python3
"""
trainer.py — Fine-tune EfficientNet-B0 for UI element classification.

Production architecture:
  1. Load cropped element images from Crops/ + labels from Dataset.csv
  2. Fine-tune EfficientNet-B0 (last 3 blocks unfrozen) with classifier head
  3. Class-weighted CrossEntropy to handle imbalance
  4. Data augmentation (color jitter, random crop, flip)
  5. Early stopping on validation loss
  6. Save fine-tuned model as arrow_ui_classifier.pt

Replaces the frozen-features + ArrowClassifier approach for production.
ArrowClassifier is great for closed-set face recognition.
For UI classification across infinite websites, fine-tuned features are better.

Usage:
    python3 trainer.py --dataset ./dataset --epochs 20
    python3 trainer.py --dataset ./dataset --epochs 30 --lr 5e-5 --batch-size 64
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UIElementDataset(Dataset):
    """Load UI element crops with labels from Dataset.csv."""

    def __init__(self, csv_path: Path, crops_dir: Path,
                 transform=None, label_encoder=None):
        self.transform = transform
        self.samples = []

        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_path = crops_dir / row["filename"]
                if img_path.exists():
                    self.samples.append({
                        "path": img_path,
                        "label": row["label"],
                    })

        if label_encoder:
            self.le = label_encoder
        else:
            self.le = LabelEncoder()
            self.le.fit([s["label"] for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_idx = self.le.transform([sample["label"]])[0]
        return img, label_idx


# ---------------------------------------------------------------------------
# Fine-tuned model
# ---------------------------------------------------------------------------

class UIClassifier(nn.Module):
    """EfficientNet-B0 with fine-tuned last blocks + classifier head."""

    def __init__(self, n_classes: int, dropout: float = 0.3, unfreeze_blocks: int = 3):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        embed_dim = self.backbone.classifier[1].in_features  # 1280

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last N blocks of backbone features
        features = list(self.backbone.features.children())
        for block in features[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_embeddings(self, x):
        """Extract 1280-D embeddings (before classifier head) for compatibility."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}: loss={loss.item():.4f}  acc={correct/total*100:.1f}%")

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 for ArrowUI.")
    parser.add_argument("--dataset", type=str, default="./dataset",
                        help="Path to dataset directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for unfrozen backbone blocks")
    parser.add_argument("--lr-head", type=float, default=1e-3,
                        help="Learning rate for classifier head")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--unfreeze-blocks", type=int, default=3,
                        help="Number of EfficientNet blocks to unfreeze (default: 3)")
    parser.add_argument("--output", type=str, default="arrow_ui_classifier.pt",
                        help="Output model path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    csv_path = dataset_dir / "Dataset.csv"
    crops_dir = dataset_dir / "Crops"

    if not csv_path.exists():
        print(f"❌ Dataset.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}")
    print(f"  ArrowUI Fine-Tuning Trainer")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_dir.resolve()}")
    print(f"  Epochs: {args.epochs}  |  LR: {args.lr}  |  LR-head: {args.lr_head}")
    print(f"  Unfreeze blocks: {args.unfreeze_blocks}")
    print(f"{'='*60}\n")

    # --- Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Load full dataset to get label encoder ---
    full_dataset = UIElementDataset(csv_path, crops_dir, transform=None)
    le = full_dataset.le
    n_classes = len(le.classes_)
    print(f"[trainer] Loaded {len(full_dataset)} samples, {n_classes} classes")
    print(f"[trainer] Classes: {list(le.classes_)}")

    # --- Split into train/val (stratified, handle rare classes) ---
    all_labels = [s["label"] for s in full_dataset.samples]
    all_indices = list(range(len(full_dataset)))
    y_encoded = le.transform(all_labels)

    class_counts = Counter(y_encoded)
    rare_mask = np.array([class_counts[y] < 5 for y in y_encoded])

    if rare_mask.any():
        rare_idx = [i for i, r in enumerate(rare_mask) if r]
        main_idx = [i for i, r in enumerate(rare_mask) if not r]
        main_y = y_encoded[np.array(main_idx)]

        train_main, val_main = train_test_split(
            main_idx, test_size=0.15, random_state=42,
            stratify=main_y
        )
        train_indices = train_main + rare_idx
        val_indices = val_main
        print(f"[trainer] {len(rare_idx)} rare-class samples added to train")
    else:
        train_indices, val_indices = train_test_split(
            all_indices, test_size=0.15, random_state=42,
            stratify=y_encoded
        )

    # --- Create split datasets ---
    class SubsetDataset(Dataset):
        def __init__(self, parent, indices, transform):
            self.parent = parent
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            sample = self.parent.samples[self.indices[idx]]
            img = Image.open(sample["path"]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label_idx = self.parent.le.transform([sample["label"]])[0]
            return img, label_idx

    train_dataset = SubsetDataset(full_dataset, train_indices, train_transform)
    val_dataset = SubsetDataset(full_dataset, val_indices, val_transform)

    print(f"[trainer] Split: {len(train_dataset)} train / {len(val_dataset)} val")

    # --- Class-weighted sampler for balanced batches ---
    train_labels = [y_encoded[i] for i in train_indices]
    class_counts_train = Counter(train_labels)
    weights = [1.0 / class_counts_train[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # --- Class-weighted loss ---
    total_train = len(train_labels)
    class_weights = torch.tensor(
        [total_train / (n_classes * class_counts_train.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Model ---
    model = UIClassifier(
        n_classes=n_classes,
        dropout=args.dropout,
        unfreeze_blocks=args.unfreeze_blocks,
    ).to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[trainer] Model params: {trainable:,} trainable / {total_params:,} total")

    # --- Optimizer: different LR for backbone vs head ---
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Training loop with early stopping ---
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    print(f"\n[trainer] Starting training...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        elapsed = time.time() - epoch_t0
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  "
              f"lr={lr_bb:.2e}/{lr_hd:.2e}  ({elapsed:.1f}s)")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_encoder_classes": list(le.classes_),
                "n_classes": n_classes,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch,
                "unfreeze_blocks": args.unfreeze_blocks,
                "dropout": args.dropout,
            }, args.output)
            print(f"         ↑ New best! Saved to {args.output}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    total_time = time.time() - t0

    # --- Load best model and final evaluation ---
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)

    # Per-class accuracy on validation set
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_preds, final_labels = evaluate(model, val_loader, criterion, device)

    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, true in zip(final_preds, final_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"  ─────────────────────────────────────")
    print(f"  Best epoch:      {best_epoch}")
    print(f"  Best val acc:    {best_val_acc*100:.2f}%")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Classes:         {n_classes}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Model size:      {Path(args.output).stat().st_size / 1024:.0f} KB")
    print(f"\n  Per-class accuracy:")
    for cls_idx in sorted(class_total.keys()):
        cls_name = le.inverse_transform([cls_idx])[0]
        total = class_total[cls_idx]
        correct = class_correct[cls_idx]
        acc = correct / total * 100 if total > 0 else 0
        bar = "█" * int(acc / 2.5)
        print(f"    {cls_name:20s}  {correct:4d}/{total:4d}  {acc:6.1f}%  {bar}")
    print(f"{'='*60}\n")

    # Also save as pickle for backward compat with classifier.py
    compat_data = {
        "model_path": args.output,
        "label_encoder_classes": list(le.classes_),
        "n_classes": n_classes,
        "train_accuracy": checkpoint["train_acc"],
        "test_accuracy": checkpoint["val_acc"],
        "class_names": list(le.classes_),
        "ood_method": "confidence_only",
        "model_type": "fine_tuned",
        "unfreeze_blocks": args.unfreeze_blocks,
        "dropout": args.dropout,
    }
    pkl_path = Path(args.output).with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(compat_data, f)
    print(f"[trainer] Metadata saved → {pkl_path}")


if __name__ == "__main__":
    main()
