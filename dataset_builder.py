#!/usr/bin/env python3
"""
dataset_builder.py — Builds training data for ArrowUI classifier.

Takes URLs (or a batch file), visits each with Playwright at the viewport
size you specify (default 1920x1080), extracts every visible element with
getBoundingClientRect(), normalizes coords to [0-1], and writes:

  - cropped element PNG (like faceR crops face images)
  - Dataset.csv mapping each crop to its canonical class label
  - Full screenshots for reference

The NORMALIZED coords make the trained model work on ANY screenshot
from ANY device at ANY resolution. No screen-size list needed.

Usage:
    python3 dataset_builder.py --url https://example.com
    python3 dataset_builder.py --batch urls.txt
    python3 dataset_builder.py --batch urls.txt --out ./my_dataset --width 1280 --height 800
    python3 dataset_builder.py --batch urls.txt --min-crops 3  # skip classes with < 3 crops
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from extractor import extract_page
from schema import canonical_class, CLASS_TO_IDX


# ---------------------------------------------------------------------------
# Crop elements from screenshot and build Dataset.csv
# ---------------------------------------------------------------------------

def crop_elements(
    page_record,
    screenshot_path: Path,
    crops_dir: Path,
    min_size: int = 8,
) -> list[dict]:
    """
    Crop each element region from the screenshot, save as individual PNGs.
    Returns list of dicts: {filename, label, class_id, text, normalized_box}

    This mirrors faceR's Faces/ folder — each crop is one training sample.
    """
    img = Image.open(screenshot_path)
    img_w, img_h = img.size
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for el in page_record.elements:
        # Map to canonical 16-class label
        label = canonical_class(el.type)
        if label is None:
            continue  # skip containers / unmapped types

        b = el.cartesian
        if b.width < min_size or b.height < min_size:
            continue

        # Pixel coords (clamped to image bounds)
        x1 = max(0, int(b.x1))
        y1 = max(0, int(b.y1))
        x2 = min(img_w, int(b.x2))
        y2 = min(img_h, int(b.y2))

        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        # Crop the element region
        crop = img.crop((x1, y1, x2, y2))

        # Unique filename: {label}_{hash}.png
        content_hash = hashlib.md5(
            f"{el.element_id}_{b.x1}_{b.y1}_{b.x2}_{b.y2}_{page_record.page_url}".encode()
        ).hexdigest()[:10]
        filename = f"{label}_{content_hash}.png"
        crop.save(str(crops_dir / filename))

        # Normalized box [0-1] — resolution independent
        norm = {
            "cx": round((b.x1 + b.x2) / 2 / img_w, 6),
            "cy": round((b.y1 + b.y2) / 2 / img_h, 6),
            "w":  round(b.width / img_w, 6),
            "h":  round(b.height / img_h, 6),
        }

        rows.append({
            "filename": filename,
            "label": label,
            "class_id": CLASS_TO_IDX[label],
            "text": el.text[:200],
            "normalized_cx": norm["cx"],
            "normalized_cy": norm["cy"],
            "normalized_w": norm["w"],
            "normalized_h": norm["h"],
            "source_url": page_record.page_url,
            "source_width": img_w,
            "source_height": img_h,
        })

    return rows


# ---------------------------------------------------------------------------
# Process one URL
# ---------------------------------------------------------------------------

def process_url(
    url: str,
    out_dir: Path,
    crops_dir: Path,
    viewport_width: int,
    viewport_height: int,
) -> list[dict]:
    """Extract elements from URL, crop them, return dataset rows."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    session_name = f"page_{url_hash}_{ts}"
    session_dir = out_dir / "screenshots" / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = session_dir / "screenshot.png"

    record = extract_page(
        url=url,
        screenshot_path=screenshot_path,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )

    rows = crop_elements(record, screenshot_path, crops_dir)

    # Save full elements JSON for reference
    elements_data = []
    for el in record.elements:
        lbl = canonical_class(el.type)
        if lbl is None:
            continue
        elements_data.append({
            "element_id": el.element_id,
            "tag": el.tag,
            "type": el.type,
            "canonical_class": lbl,
            "text": el.text[:200],
            "cartesian": el.cartesian.model_dump(),
            "is_interactive": el.is_interactive,
        })

    with open(session_dir / "elements.json", "w", encoding="utf-8") as f:
        json.dump({
            "url": url,
            "viewport": {"width": viewport_width, "height": viewport_height},
            "total_classified_elements": len(elements_data),
            "elements": elements_data,
        }, f, indent=2, ensure_ascii=False)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build ArrowUI training dataset from web pages.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url",   type=str, help="Single URL")
    source.add_argument("--batch", type=str, help="Text file with one URL per line")

    parser.add_argument("--out", type=str, default="./dataset", help="Output dir (default: ./dataset)")
    parser.add_argument("--width", type=int, default=1920, help="Viewport width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Viewport height (default: 1080)")
    parser.add_argument("--min-crops", type=int, default=1, help="Skip classes with fewer crops than this")
    parser.add_argument("--multi-viewport", action="store_true",
                        help="Scrape each URL at 3 viewports: mobile(375x667), tablet(768x1024), desktop(1920x1080)")

    args = parser.parse_args()
    out_dir = Path(args.out)
    crops_dir = out_dir / "Crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Collect URLs
    if args.url:
        urls = [args.url.strip()]
    else:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"❌ Batch file not found: {batch_path}", file=sys.stderr)
            sys.exit(1)
        urls = [u.strip() for u in batch_path.read_text().splitlines()
                if u.strip() and not u.startswith("#")]

    # Multi-viewport: each URL scraped at 3 sizes
    VIEWPORTS = [
        (375, 667, "mobile"),
        (768, 1024, "tablet"),
        (1920, 1080, "desktop"),
    ] if args.multi_viewport else [(args.width, args.height, "single")]

    total_jobs = len(urls) * len(VIEWPORTS)
    print(f"\n{'='*60}")
    print(f"  ArrowUI Dataset Builder")
    print(f"  URLs: {len(urls)}  |  Viewports: {len(VIEWPORTS)} {'(mobile/tablet/desktop)' if args.multi_viewport else f'({args.width}×{args.height})'}")
    print(f"  Total scrape jobs: {total_jobs}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"{'='*60}\n")

    all_rows = []
    job = 0
    for i, url in enumerate(urls, 1):
        for vw, vh, vname in VIEWPORTS:
            job += 1
            label = f"[{job}/{total_jobs}]" if args.multi_viewport else f"[{i}/{len(urls)}]"
            print(f"{label} {url} ({vname} {vw}×{vh})")
            try:
                rows = process_url(url, out_dir, crops_dir, vw, vh)
                all_rows.extend(rows)
                print(f"         → {len(rows)} element crops saved")
            except Exception as exc:
                print(f"         ❌ Failed: {exc}", file=sys.stderr)

    # Write Dataset.csv (mirrors faceR's Dataset.csv)
    csv_path = out_dir / "Dataset.csv"
    fieldnames = [
        "filename", "label", "class_id", "text",
        "normalized_cx", "normalized_cy", "normalized_w", "normalized_h",
        "source_url", "source_width", "source_height",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print class distribution
    from collections import Counter
    dist = Counter(r["label"] for r in all_rows)
    print(f"\n{'='*60}")
    print(f"  Dataset complete: {len(all_rows)} total crops")
    print(f"  Saved to: {csv_path}")
    print(f"\n  Class distribution:")
    for cls, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 40)
        print(f"    {cls:20s}  {count:5d}  {bar}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

