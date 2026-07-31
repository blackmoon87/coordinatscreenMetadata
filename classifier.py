#!/usr/bin/env python3
"""
classifier.py — ArrowUI inference pipeline.

Input:  ANY screenshot (any resolution, any device)
Output: JSON with detected UI elements, their types, text, confidence,
        and NORMALIZED coordinates [0-1] that work on any screen size.

Architecture mirrors faceR's detect.py / FaceRecognizerPipeline:
  1. Slide a multi-scale window over the screenshot
  2. Extract 1280-D EfficientNet-B0 embedding per region
  3. ArrowClassifier predicts type + OOD rejection
  4. Non-max suppression to remove overlapping boxes
  5. Output normalized coords + type + confidence

Usage:
    python3 classifier.py --image screenshot.png
    python3 classifier.py --image screenshot.png --output results.json
    python3 classifier.py --image screenshot.png --annotate annotated.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models

# Add parent directory for ArrowClassifier
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))

from schema import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NormalizedBox, DetectionResult


# ---------------------------------------------------------------------------
# Sliding window region proposals (replaces MTCNN face detector)
# ---------------------------------------------------------------------------

def generate_proposals(
    img_w: int,
    img_h: int,
    min_size: int = 20,
    scales: list[float] | None = None,
    aspect_ratios: list[float] | None = None,
    stride_ratio: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """
    Generate multi-scale region proposals across the image.
    Returns list of (x1, y1, x2, y2) pixel boxes.

    Unlike MTCNN which only finds faces, this generates candidate regions
    at multiple scales and aspect ratios to catch all UI element types.
    """
    if scales is None:
        # Proportional scales — work at any resolution
        scales = [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48]
    if aspect_ratios is None:
        # UI elements come in many shapes: wide buttons, tall lists, square icons
        aspect_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]

    proposals = []
    for scale in scales:
        base_size = int(min(img_w, img_h) * scale)
        if base_size < min_size:
            continue

        for ar in aspect_ratios:
            # Width and height from scale + aspect ratio
            w = int(base_size * np.sqrt(ar))
            h = int(base_size / np.sqrt(ar))
            if w < min_size or h < min_size:
                continue
            if w > img_w or h > img_h:
                continue

            stride_x = max(min_size, int(w * stride_ratio))
            stride_y = max(min_size, int(h * stride_ratio))

            for y in range(0, img_h - h + 1, stride_y):
                for x in range(0, img_w - w + 1, stride_x):
                    proposals.append((x, y, x + w, y + h))

    return proposals


# ---------------------------------------------------------------------------
# Selective proposals (faster — uses edge/contrast detection)
# ---------------------------------------------------------------------------

def generate_selective_proposals(
    img: Image.Image,
    max_proposals: int = 2000,
    min_size: int = 16,
) -> list[tuple[int, int, int, int]]:
    """
    Smart region proposals using contrast/edge detection.
    Much faster than exhaustive sliding window — focuses on areas with visual content.
    """
    img_w, img_h = img.size

    # Convert to grayscale numpy
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Simple edge detection: absolute gradient
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))

    # Pad to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant")
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")
    edge_map = grad_x + grad_y

    # Find high-gradient regions using a coarse grid
    proposals = []
    cell_sizes = [16, 32, 48, 64, 96, 128, 192]

    for cell in cell_sizes:
        for y in range(0, img_h - cell + 1, cell // 2):
            for x in range(0, img_w - cell + 1, cell // 2):
                region = edge_map[y:y+cell, x:x+cell]
                if region.mean() > 5.0:  # has visual content
                    # Try multiple aspect ratios around this point
                    for w_mult, h_mult in [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3)]:
                        bw = min(cell * w_mult, img_w - x)
                        bh = min(cell * h_mult, img_h - y)
                        if bw >= min_size and bh >= min_size:
                            proposals.append((x, y, x + bw, y + bh))

    # Deduplicate close proposals
    if len(proposals) > max_proposals:
        # Prioritize by edge density
        scores = []
        for x1, y1, x2, y2 in proposals:
            region = edge_map[y1:y2, x1:x2]
            scores.append(region.mean())
        # Keep top-K by edge density
        indices = np.argsort(scores)[-max_proposals:]
        proposals = [proposals[i] for i in indices]

    return proposals


# ---------------------------------------------------------------------------
# Non-max suppression
# ---------------------------------------------------------------------------

def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.4,
) -> np.ndarray:
    """Standard NMS. Returns indices to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# ArrowUI Pipeline (mirrors FaceRecognizerPipeline from detect.py)
# ---------------------------------------------------------------------------

class ArrowUIClassifier:
    """
    End-to-end UI element classifier.
    Input: screenshot image (any size)
    Output: list of DetectionResult with normalized coords.

    Supports two model formats:
      - Fine-tuned: .pt file (EfficientNet-B0 fine-tuned end-to-end)
      - Legacy: .pkl file (frozen EfficientNet + ArrowClassifier)
    """

    def __init__(
        self,
        model_path: str = "arrow_ui_classifier.pt",
        confidence_threshold: float = 0.60,
        iou_threshold: float = 0.4,
        max_proposals: int = 2000,
        device: torch.device | None = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_proposals = max_proposals
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        model_path = Path(model_path)

        # --- Load fine-tuned model (.pt) ---
        if model_path.suffix == ".pt" and model_path.exists():
            from trainer import UIClassifier
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.class_names = checkpoint["label_encoder_classes"]
            n_classes = checkpoint["n_classes"]
            self.model_type = "fine_tuned"

            self.model = UIClassifier(
                n_classes=n_classes,
                dropout=checkpoint.get("dropout", 0.3),
                unfreeze_blocks=checkpoint.get("unfreeze_blocks", 3),
            ).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            print(f"[classifier] Fine-tuned model loaded: {n_classes} classes, "
                  f"val_acc={checkpoint.get('val_acc', 0)*100:.1f}%")

        # --- Fallback: legacy ArrowClassifier (.pkl) ---
        elif model_path.with_suffix(".pkl").exists():
            with open(model_path.with_suffix(".pkl"), "rb") as f:
                data = pickle.load(f)
            self.classifier = data["classifier"]
            self.label_encoder = data["label_encoder"]
            self.class_names = data.get("class_names", list(self.label_encoder.classes_))
            self.model_type = "legacy"

            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone.classifier = torch.nn.Identity()
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()

            print(f"[classifier] Legacy model loaded: {len(self.class_names)} classes")
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {model_path.with_suffix('.pkl')}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _batch_extract(self, crops: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Extract embeddings for multiple crops in batches."""
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            tensors = torch.stack([self.transform(c.convert("RGB")) for c in batch]).to(self.device)
            with torch.no_grad():
                if self.model_type == "fine_tuned":
                    embeddings = self.model.get_embeddings(tensors)
                else:
                    embeddings = self.backbone(tensors)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 1280))

    def _batch_classify(self, crops: list[Image.Image], batch_size: int = 64):
        """Classify crops and return labels + confidence scores."""
        all_labels = []
        all_probs = []

        for i in range(0, len(crops), batch_size):
            batch = crops[i:i+batch_size]
            tensors = torch.stack([self.transform(c.convert("RGB")) for c in batch]).to(self.device)

            with torch.no_grad():
                if self.model_type == "fine_tuned":
                    logits = self.model(tensors)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    max_probs, pred_indices = probs.max(dim=1)
                    labels = [self.class_names[idx] for idx in pred_indices.cpu().numpy()]
                else:
                    embeddings = self.backbone(tensors).cpu().numpy()
                    clf_probs = self.classifier.predict_proba(embeddings, temperature=5.0)
                    max_probs_np = np.max(clf_probs, axis=1)
                    pred_indices = self.classifier.predict(embeddings)
                    labels = list(self.label_encoder.inverse_transform(pred_indices))
                    max_probs = torch.tensor(max_probs_np)

                all_labels.extend(labels)
                all_probs.extend(max_probs.cpu().numpy())

        return all_labels, np.array(all_probs)

    def predict(self, image: Image.Image | str | Path) -> list[DetectionResult]:
        """
        Run full inference on a screenshot.

        Args:
            image: PIL Image, or path to screenshot file (any resolution).

        Returns:
            List of DetectionResult with normalized coords [0-1].
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        img_w, img_h = image.size
        t0 = time.time()

        # 1. Generate region proposals
        proposals = generate_selective_proposals(image, max_proposals=self.max_proposals)
        if not proposals:
            print("[classifier] No proposals generated.")
            return []
        print(f"[classifier] {len(proposals)} region proposals ({img_w}×{img_h})")

        # 2. Crop each proposal region
        crops = []
        valid_proposals = []
        for (x1, y1, x2, y2) in proposals:
            crop = image.crop((x1, y1, x2, y2))
            if crop.size[0] >= 8 and crop.size[1] >= 8:
                crops.append(crop)
                valid_proposals.append((x1, y1, x2, y2))

        if not crops:
            return []

        # 3. Classify all crops — confidence-only filtering
        pred_labels, max_probs = self._batch_classify(crops)

        # 4. Collect results above confidence threshold
        boxes = []
        scores = []
        labels = []
        results_raw = []

        for i, (pred, prob) in enumerate(zip(pred_labels, max_probs)):
            if prob < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = valid_proposals[i]
            boxes.append([x1, y1, x2, y2])
            scores.append(float(prob))
            labels.append(pred)
            results_raw.append(i)

        if not boxes:
            print("[classifier] All proposals rejected as OOD.")
            return []

        boxes_arr = np.array(boxes, dtype=float)
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # 6. Non-max suppression
        keep = nms(boxes_arr, scores_arr, labels_arr, iou_threshold=self.iou_threshold)

        # 7. Build output with normalized coords
        results = []
        for idx, k in enumerate(keep):
            x1, y1, x2, y2 = boxes_arr[k]
            norm = NormalizedBox(
                cx=round(((x1 + x2) / 2) / img_w, 6),
                cy=round(((y1 + y2) / 2) / img_h, 6),
                w=round((x2 - x1) / img_w, 6),
                h=round((y2 - y1) / img_h, 6),
            )
            label = labels_arr[k]
            class_id = CLASS_TO_IDX.get(label, -1)

            results.append(DetectionResult(
                element_id=f"el_{idx:04d}",
                type=label,
                class_id=class_id,
                text="",  # text extraction requires OCR — separate step
                confidence=round(scores_arr[k], 4),
                is_ood=False,
                normalized=norm,
            ))

        elapsed = time.time() - t0
        print(f"[classifier] {len(results)} elements detected in {elapsed:.2f}s")
        return results

    def predict_json(self, image: Image.Image | str | Path) -> dict:
        """Run prediction and return full JSON output."""
        if isinstance(image, (str, Path)):
            img_path = str(image)
            image = Image.open(image).convert("RGB")
        else:
            img_path = "<in-memory>"

        results = self.predict(image)
        img_w, img_h = image.size

        return {
            "source": img_path,
            "source_width": img_w,
            "source_height": img_h,
            "total_detections": len(results),
            "elements": [
                {
                    "element_id": r.element_id,
                    "type": r.type,
                    "class_id": r.class_id,
                    "text": r.text,
                    "confidence": r.confidence,
                    "normalized": r.normalized.model_dump(),
                    "pixels": r.normalized.to_pixels(img_w, img_h),
                }
                for r in results
            ],
        }

    def annotate(self, image: Image.Image | str | Path,
                 output_path: str | Path) -> Path:
        """Run prediction and draw annotated bounding boxes."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        results = self.predict(image)
        img_w, img_h = image.size
        draw = ImageDraw.Draw(image)

        TYPE_COLORS = {
            "button": (50, 205, 50), "link": (30, 144, 255),
            "input_text": (255, 140, 0), "input_checkbox": (218, 112, 214),
            "input_radio": (218, 112, 214), "dropdown": (255, 215, 0),
            "textarea": (255, 140, 0), "image": (147, 112, 219),
            "heading": (255, 69, 0), "text": (169, 169, 169),
            "list": (95, 158, 160), "list_item": (95, 158, 160),
            "navigation": (64, 224, 208), "form": (255, 99, 71),
            "table": (60, 179, 113), "media": (100, 149, 237),
        }

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            font = ImageFont.load_default()

        for r in results:
            px = r.normalized.to_pixels(img_w, img_h)
            color = TYPE_COLORS.get(r.type, (200, 200, 200))

            # Draw box
            for offset in range(2):
                draw.rectangle(
                    [px["x1"] + offset, px["y1"] + offset,
                     px["x2"] - offset, px["y2"] - offset],
                    outline=color,
                )

            # Label
            label = f"{r.type} {r.confidence:.0%}"
            bbox = draw.textbbox((px["x1"], px["y1"] - 16), label, font=font)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1],
                           fill=(0, 0, 0))
            draw.text((px["x1"], px["y1"] - 16), label, fill=color, font=font)

        output_path = Path(output_path)
        image.save(str(output_path))
        print(f"[classifier] Annotated image saved → {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ArrowUI — Classify UI elements in screenshots.")
    parser.add_argument("--image", type=str, required=True, help="Path to screenshot (any resolution)")
    parser.add_argument("--model", type=str, default="arrow_ui_classifier.pkl", help="Model path")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--annotate", type=str, default=None, help="Save annotated image")
    parser.add_argument("--confidence", type=float, default=0.50, help="Min confidence (default: 0.50)")
    parser.add_argument("--ood-threshold", type=float, default=-0.475, help="OOD score threshold")
    args = parser.parse_args()

    pipe = ArrowUIClassifier(
        model_path=args.model,
        confidence_threshold=args.confidence,
        ood_score_threshold=args.ood_threshold,
    )

    # Run inference
    result = pipe.predict_json(args.image)

    # Print results
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save JSON
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[classifier] JSON saved → {args.output}")

    # Save annotated image
    if args.annotate:
        pipe.annotate(args.image, args.annotate)


if __name__ == "__main__":
    main()
