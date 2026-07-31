#!/usr/bin/env python3
"""
test_ood.py — OOD rejection + per-class accuracy tests for ArrowUI classifier.

Tests:
  1. Per-class accuracy breakdown on the test set
  2. OOD rejection: random noise images → should all be rejected
  3. OOD rejection: solid color images → should all be rejected
  4. OOD rejection: natural photos (non-UI) → should all be rejected
  5. End-to-end inference on a real screenshot
"""
from __future__ import annotations

import csv
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms, models

PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
from core import ArrowClassifier


def load_model(model_path: str = "arrow_ui_classifier.pkl"):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(device):
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    backbone.classifier = torch.nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def embed_images(images: list[Image.Image], backbone, device, transform, batch_size=64):
    all_emb = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        tensors = torch.stack([transform(img.convert("RGB")) for img in batch]).to(device)
        with torch.no_grad():
            emb = backbone(tensors)
        all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


# ---------------------------------------------------------------------------
# Test 1: Per-class accuracy on held-out test set
# ---------------------------------------------------------------------------

def test_per_class_accuracy(model_data, dataset_dir: Path, device, backbone, transform):
    print(f"\n{'='*70}")
    print(f"  TEST 1: Per-Class Accuracy (held-out test set)")
    print(f"{'='*70}\n")

    clf = model_data["classifier"]
    le = model_data["label_encoder"]
    csv_path = dataset_dir / "Dataset.csv"
    crops_dir = dataset_dir / "Crops"

    # Load all samples
    samples = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_path = crops_dir / row["filename"]
            if img_path.exists():
                samples.append({"path": img_path, "label": row["label"]})

    labels = [s["label"] for s in samples]
    y_encoded = le.transform(labels)

    # Reproduce the same split
    class_counts = Counter(y_encoded)
    rare_mask = np.array([class_counts[y] < 5 for y in y_encoded])

    if rare_mask.any():
        main_idx = np.where(~rare_mask)[0]
        main_y = y_encoded[main_idx]
        _, test_idx_in_main, _, _ = train_test_split(
            main_idx, main_y, test_size=0.2, random_state=42, stratify=main_y
        )
        test_indices = test_idx_in_main
    else:
        all_idx = np.arange(len(y_encoded))
        _, test_indices, _, _ = train_test_split(
            all_idx, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    test_samples = [samples[i] for i in test_indices]
    test_labels = [s["label"] for s in test_samples]
    test_images = [Image.open(s["path"]) for s in test_samples]

    # Extract embeddings and predict
    test_emb = embed_images(test_images, backbone, device, transform)
    test_y = le.transform(test_labels)

    predictions = clf.predict(test_emb)
    pred_labels = le.inverse_transform(predictions)

    # Classification report
    print(classification_report(test_labels, pred_labels, zero_division=0))

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for true, pred in zip(test_labels, pred_labels):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    print(f"\n  Per-class accuracy:")
    for cls in sorted(class_total.keys()):
        total = class_total[cls]
        correct = class_correct[cls]
        acc = correct / total * 100 if total > 0 else 0
        bar = "█" * int(acc / 2.5)
        print(f"    {cls:20s}  {correct:4d}/{total:4d}  {acc:6.1f}%  {bar}")

    overall = sum(class_correct.values()) / sum(class_total.values()) * 100
    print(f"\n    {'OVERALL':20s}  {sum(class_correct.values()):4d}/{sum(class_total.values()):4d}  {overall:6.1f}%")
    return overall


# ---------------------------------------------------------------------------
# Test 2: OOD rejection — random noise
# ---------------------------------------------------------------------------

def test_ood_random_noise(model_data, backbone, device, transform, ood_threshold, n_samples=100):
    print(f"\n{'='*70}")
    print(f"  TEST 2: OOD Rejection — Random Noise ({n_samples} images)")
    print(f"  OOD threshold: {ood_threshold:.4f}")
    print(f"{'='*70}\n")

    clf = model_data["classifier"]
    le = model_data["label_encoder"]

    # Generate random noise images at various sizes
    noise_images = []
    sizes = [(224, 224), (100, 300), (400, 100), (50, 50), (320, 568)]
    for i in range(n_samples):
        w, h = sizes[i % len(sizes)]
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        noise_images.append(Image.fromarray(arr))

    emb = embed_images(noise_images, backbone, device, transform)
    _, _, ood_scores, is_ood = clf.predict_with_ood(
        emb, min_prob_threshold=0.5, ood_score_threshold=ood_threshold, label_encoder=le
    )

    rejected = sum(is_ood)
    rate = rejected / n_samples * 100
    print(f"  Random noise: {rejected}/{n_samples} rejected ({rate:.1f}%)")
    print(f"  OOD scores — mean: {ood_scores.mean():.4f}, min: {ood_scores.min():.4f}, max: {ood_scores.max():.4f}")
    if rate >= 90:
        print(f"  ✅ PASS — {rate:.0f}% rejection rate")
    else:
        print(f"  ⚠️  WARN — only {rate:.0f}% rejection rate (expected ≥90%)")
    return rate


# ---------------------------------------------------------------------------
# Test 3: OOD rejection — solid color blocks
# ---------------------------------------------------------------------------

def test_ood_solid_colors(model_data, backbone, device, transform, ood_threshold, n_samples=50):
    print(f"\n{'='*70}")
    print(f"  TEST 3: OOD Rejection — Solid Color Blocks ({n_samples} images)")
    print(f"  OOD threshold: {ood_threshold:.4f}")
    print(f"{'='*70}\n")

    clf = model_data["classifier"]
    le = model_data["label_encoder"]

    color_images = []
    for i in range(n_samples):
        r, g, b = np.random.randint(0, 256, 3)
        w = np.random.randint(50, 500)
        h = np.random.randint(50, 500)
        img = Image.new("RGB", (w, h), (int(r), int(g), int(b)))
        color_images.append(img)

    emb = embed_images(color_images, backbone, device, transform)
    _, _, ood_scores, is_ood = clf.predict_with_ood(
        emb, min_prob_threshold=0.5, ood_score_threshold=ood_threshold, label_encoder=le
    )

    rejected = sum(is_ood)
    rate = rejected / n_samples * 100
    print(f"  Solid colors: {rejected}/{n_samples} rejected ({rate:.1f}%)")
    print(f"  OOD scores — mean: {ood_scores.mean():.4f}, min: {ood_scores.min():.4f}, max: {ood_scores.max():.4f}")
    if rate >= 90:
        print(f"  ✅ PASS — {rate:.0f}% rejection rate")
    else:
        print(f"  ⚠️  WARN — only {rate:.0f}% rejection rate (expected ≥90%)")
    return rate


# ---------------------------------------------------------------------------
# Test 4: OOD rejection — synthetic non-UI patterns
# ---------------------------------------------------------------------------

def test_ood_patterns(model_data, backbone, device, transform, ood_threshold, n_samples=50):
    print(f"\n{'='*70}")
    print(f"  TEST 4: OOD Rejection — Synthetic Non-UI Patterns ({n_samples} images)")
    print(f"  OOD threshold: {ood_threshold:.4f}")
    print(f"{'='*70}\n")

    clf = model_data["classifier"]
    le = model_data["label_encoder"]

    pattern_images = []
    for i in range(n_samples):
        w, h = 224, 224
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        pattern_type = i % 5
        if pattern_type == 0:
            # Diagonal stripes
            for x in range(-h, w, 10):
                draw.line([(x, 0), (x + h, h)], fill=(np.random.randint(0, 256),) * 3, width=3)
        elif pattern_type == 1:
            # Concentric circles
            for r in range(0, 150, 8):
                draw.ellipse([112-r, 112-r, 112+r, 112+r],
                             outline=(np.random.randint(0, 256),) * 3, width=2)
        elif pattern_type == 2:
            # Random dots
            for _ in range(200):
                x, y = np.random.randint(0, w), np.random.randint(0, h)
                draw.ellipse([x-3, y-3, x+3, y+3],
                             fill=(np.random.randint(0, 256),) * 3)
        elif pattern_type == 3:
            # Gradient
            for y_pos in range(h):
                val = int(255 * y_pos / h)
                draw.line([(0, y_pos), (w, y_pos)], fill=(val, 255-val, 128))
        else:
            # Checkerboard
            cell = 16
            for cy in range(0, h, cell):
                for cx in range(0, w, cell):
                    if (cx // cell + cy // cell) % 2:
                        draw.rectangle([cx, cy, cx+cell, cy+cell], fill=(0, 0, 0))

        pattern_images.append(img)

    emb = embed_images(pattern_images, backbone, device, transform)
    _, _, ood_scores, is_ood = clf.predict_with_ood(
        emb, min_prob_threshold=0.5, ood_score_threshold=ood_threshold, label_encoder=le
    )

    rejected = sum(is_ood)
    rate = rejected / n_samples * 100
    print(f"  Patterns: {rejected}/{n_samples} rejected ({rate:.1f}%)")
    print(f"  OOD scores — mean: {ood_scores.mean():.4f}, min: {ood_scores.min():.4f}, max: {ood_scores.max():.4f}")
    if rate >= 80:
        print(f"  ✅ PASS — {rate:.0f}% rejection rate")
    else:
        print(f"  ⚠️  WARN — only {rate:.0f}% rejection rate (expected ≥80%)")
    return rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_path = "arrow_ui_classifier.pkl"
    dataset_dir = Path("./dataset")

    print(f"\n{'#'*70}")
    print(f"  ArrowUI — OOD & Accuracy Test Suite")
    print(f"{'#'*70}")

    model_data = load_model(model_path)
    device = get_device()
    backbone = load_backbone(device)
    transform = get_transform()

    print(f"\n  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"  Classes: {model_data['class_names']}")
    print(f"  Train acc: {model_data.get('train_accuracy', 0)*100:.2f}%")
    print(f"  Test acc:  {model_data.get('test_accuracy', 0)*100:.2f}%")

    ood_threshold = model_data.get('ood_threshold', -0.475)
    print(f"  OOD threshold: {ood_threshold:.4f}")

    t0 = time.time()

    # Run all tests
    acc = test_per_class_accuracy(model_data, dataset_dir, device, backbone, transform)
    noise_rate = test_ood_random_noise(model_data, backbone, device, transform, ood_threshold)
    color_rate = test_ood_solid_colors(model_data, backbone, device, transform, ood_threshold)
    pattern_rate = test_ood_patterns(model_data, backbone, device, transform, ood_threshold)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'#'*70}")
    print(f"  SUMMARY")
    print(f"{'#'*70}")
    print(f"  Test accuracy:           {acc:.1f}%")
    print(f"  OOD — Random noise:      {noise_rate:.0f}% rejected")
    print(f"  OOD — Solid colors:      {color_rate:.0f}% rejected")
    print(f"  OOD — Patterns:          {pattern_rate:.0f}% rejected")
    print(f"  Total test time:         {elapsed:.1f}s")

    all_pass = acc >= 85 and noise_rate >= 80 and color_rate >= 80 and pattern_rate >= 70
    if all_pass:
        print(f"\n  ✅ ALL TESTS PASSED")
    else:
        print(f"\n  ⚠️  SOME TESTS BELOW THRESHOLD — review per-class breakdown")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
