#!/usr/bin/env python3
"""
click_by_model_eyes.py — Visual Click Automation driven by ArrowUI model's eyes.

How it works:
  1. Playwright opens ANY web page / application
  2. Takes a screenshot of the screen
  3. ArrowUIClassifier inspects the screenshot using neural vision (16 UI classes)
  4. Model outputs normalized Cartesian coordinates [cx, cy, w, h] in range [0.0, 1.0]
  5. Script converts normalized coords → pixel coords for the current screen size:
        pixel_x = int(cx * viewport_width)
        pixel_y = int(cy * viewport_height)
  6. Playwright mouse moves to (pixel_x, pixel_y) and clicks!
  7. Draws visual click verification markers on annotated screenshot.

Usage:
    python3 click_by_model_eyes.py --url https://the-internet.herokuapp.com/login --target button
    python3 click_by_model_eyes.py --url https://github.com/login --target input_text
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

# Import ArrowUI classifier
from classifier import ArrowUIClassifier


def click_element_by_model_eyes(
    url: str,
    target_type: str = "button",
    target_text_search: str | None = None,
    model_path: str = "arrow_ui_classifier.pt",
    viewport_w: int = 1280,
    viewport_h: int = 800,
    output_dir: str = "./click_verification",
) -> dict:
    """
    Open URL, inspect screen with ArrowUI model, locate target UI element, and click it.

    Args:
        url: Page URL to open.
        target_type: Type of element to click ('button', 'link', 'input_text', etc.).
        target_text_search: Optional text filter.
        model_path: Path to trained ArrowUI model (.pt or .pkl).
        viewport_w: Viewport width.
        viewport_h: Viewport height.
        output_dir: Directory to save visual verification screenshots.

    Returns:
        Dict with click target coordinates and operation status.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load ArrowUI Vision Model
    print(f"\n[Model Eyes] Loading AI vision model: {model_path}...")
    classifier = ArrowUIClassifier(model_path=model_path, confidence_threshold=0.50)

    with sync_playwright() as p:
        print(f"[Browser] Launching browser (viewport: {viewport_w}×{viewport_h})...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})

        print(f"[Browser] Navigating to {url}...")
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(1)

        # Step 1: Capture screenshot
        screenshot_before = out_path / "1_before_click.png"
        page.screenshot(path=str(screenshot_before))
        print(f"[Model Eyes] Captured screen image → {screenshot_before}")

        # Step 2: Pass screenshot to ArrowUI model's "eyes"
        print(f"[Model Eyes] Inspecting screen with neural network...")
        t0 = time.time()
        detections = classifier.predict(screenshot_before)
        elapsed = time.time() - t0
        print(f"[Model Eyes] Detected {len(detections)} UI elements in {elapsed:.2f}s")

        # Step 3: Filter for target type
        target_candidates = [d for d in detections if d.type == target_type]

        if not target_candidates:
            print(f"⚠️  No elements of type '{target_type}' detected. Available types: "
                  f"{sorted(set(d.type for d in detections))}")
            # Fall back to highest confidence interactive element if requested target not found
            interactive = [d for d in detections if d.type in ("button", "link", "input_text", "dropdown")]
            if interactive:
                target_element = interactive[0]
                print(f"  ↪ Falling back to nearest interactive element: '{target_element.type}'")
            else:
                browser.close()
                return {"status": "failed", "reason": f"No {target_type} found"}
        else:
            # Pick candidate with highest confidence
            target_element = max(target_candidates, key=lambda d: d.confidence)

        # Step 4: Calculate exact pixel click coordinates from model's normalized Cartesian coords [0-1]
        norm = target_element.normalized
        click_px_x = int(norm.cx * viewport_w)
        click_px_y = int(norm.cy * viewport_h)

        print(f"\n============================================================")
        print(f"  TARGET FOUND BY MODEL EYES")
        print(f"============================================================")
        print(f"  Element Type:     {target_element.type} (class_id: {target_element.class_id})")
        print(f"  Model Confidence: {target_element.confidence * 100:.1f}%")
        print(f"  Normalized Coords: cx={norm.cx:.4f}, cy={norm.cy:.4f}, w={norm.w:.4f}, h={norm.h:.4f}")
        print(f"  Calculated Click:  X={click_px_x}px, Y={click_px_y}px")
        print(f"============================================================\n")

        # Step 5: Execute click at model's calculated coordinates
        print(f"[Action] Moving mouse to ({click_px_x}, {click_px_y}) and clicking...")
        page.mouse.move(click_px_x, click_px_y)
        time.sleep(0.3)
        page.mouse.click(click_px_x, click_px_y)
        time.sleep(1.5)

        # Step 6: Capture post-click screenshot
        screenshot_after = out_path / "2_after_click.png"
        page.screenshot(path=str(screenshot_after))
        print(f"[Action] Post-click screenshot saved → {screenshot_after}")

        # Step 7: Draw visual verification overlay (target box + red crosshair target)
        img_verify = Image.open(screenshot_before).convert("RGB")
        draw = ImageDraw.Draw(img_verify)

        # Draw all detected boxes in light gray
        for det in detections:
            px = det.normalized.to_pixels(viewport_w, viewport_h)
            draw.rectangle([px["x1"], px["y1"], px["x2"], px["y2"]], outline=(180, 180, 180), width=1)

        # Highlight target box in bright neon green
        t_px = norm.to_pixels(viewport_w, viewport_h)
        draw.rectangle([t_px["x1"], t_px["y1"], t_px["x2"], t_px["y2"]], outline=(0, 255, 0), width=3)

        # Draw red bullseye target at exact click point (click_px_x, click_px_y)
        r = 12
        draw.ellipse([click_px_x - r, click_px_y - r, click_px_x + r, click_px_y + r], outline=(255, 0, 0), width=3)
        draw.line([click_px_x - r * 2, click_px_y, click_px_x + r * 2, click_px_y], fill=(255, 0, 0), width=2)
        draw.line([click_px_x, click_px_y - r * 2, click_px_y, click_px_y + r * 2], fill=(255, 0, 0), width=2)

        # Add label
        label_str = f"CLICKED: {target_element.type} ({target_element.confidence*100:.0f}%) @ ({click_px_x}, {click_px_y})"
        draw.rectangle([click_px_x - 10, click_px_y - 35, click_px_x + 320, click_px_y - 10], fill=(255, 0, 0))
        draw.text((click_px_x - 5, click_px_y - 32), label_str, fill=(255, 255, 255))

        verify_path = out_path / "3_model_eyes_click_verification.png"
        img_verify.save(verify_path)
        print(f"✅ Visual verification overlay saved → {verify_path}")

        browser.close()

        return {
            "status": "success",
            "element_type": target_element.type,
            "confidence": target_element.confidence,
            "normalized_coords": {"cx": norm.cx, "cy": norm.cy, "w": norm.w, "h": norm.h},
            "click_pixels": {"x": click_px_x, "y": click_px_y},
            "verification_image": str(verify_path),
        }


def main():
    parser = argparse.ArgumentParser(description="Click web elements using ArrowUI model's eyes.")
    parser.add_argument("--url", type=str, default="https://the-internet.herokuapp.com/login",
                        help="Target web URL")
    parser.add_argument("--target", type=str, default="button",
                        help="Target element type to click (e.g. button, input_text, link)")
    parser.add_argument("--model", type=str, default="arrow_ui_classifier.pt",
                        help="Model checkpoint path")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width")
    parser.add_argument("--height", type=int, default=800, help="Viewport height")
    args = parser.parse_args()

    click_element_by_model_eyes(
        url=args.url,
        target_type=args.target,
        model_path=args.model,
        viewport_w=args.width,
        viewport_h=args.height,
    )


if __name__ == "__main__":
    main()
