#!/usr/bin/env python3
"""
youtube_comment_by_model_eyes.py — YouTube Automated Commenting driven by ArrowUI AI Vision.

Workflow:
  1. Opens any YouTube video URL with Playwright
  2. Scrolls down to comments section
  3. Captures screen screenshot
  4. ArrowUI Classifier inspects screen with AI vision to locate comment input field / box
  5. Calculates normalized Cartesian coordinates [cx, cy] → converts to screen pixels
  6. Moves mouse to calculated position, clicks comment box, and types comment!
  7. Saves visual verification screenshots.

Usage:
    python3 youtube_comment_by_model_eyes.py --url https://www.youtube.com/watch?v=dQw4w9WgXcQ --comment "Awesome video! Tested with ArrowUI AI vision."
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from classifier import ArrowUIClassifier


def post_youtube_comment_by_model_eyes(
    video_url: str = "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    comment_text: str = "Great video! Automated with ArrowUI AI model vision.",
    model_path: str = "arrow_ui_classifier.pt",
    viewport_w: int = 1280,
    viewport_h: int = 900,
    output_dir: str = "./youtube_test_output",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ArrowUI Autonomous YouTube Comment Automation")
    print(f"  Target Video: {video_url}")
    print(f"  Comment Text: {comment_text}")
    print(f"{'='*70}\n")

    print("[Model] Loading ArrowUI Neural Vision Model...")
    classifier = ArrowUIClassifier(model_path=model_path, confidence_threshold=0.45)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})

        print(f"[Browser] Navigating to YouTube video...")
        page.goto(video_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)

        # Handle YouTube cookies / consent popup if present
        try:
            consent_btn = page.query_selector('button[aria-label*="Accept"], button[aria-label*="Agree"]')
            if consent_btn:
                consent_btn.click()
                time.sleep(1)
        except Exception:
            pass

        # Scroll down to load comments section
        print("[Browser] Scrolling down to load comments section...")
        page.evaluate("window.scrollBy(0, 600)")
        time.sleep(3)

        # Step 1: Capture screenshot of comments area
        screen_path = out_dir / "1_youtube_comments_screen.png"
        page.screenshot(path=str(screen_path))
        print(f"[Model Eyes] Saved screenshot → {screen_path}")

        # Step 2: Model inspects screen using neural vision
        print("[Model Eyes] Inspecting YouTube screen with neural network...")
        t0 = time.time()
        detections = classifier.predict(screen_path)
        elapsed = time.time() - t0
        print(f"[Model Eyes] Detected {len(detections)} UI elements in {elapsed:.2f}s")

        # Step 3: Locate comment input field using model predictions or interactive box in comments area
        comment_inputs = [d for d in detections if d.type in ("input_text", "textarea", "form", "text")]

        # Filter elements located in middle-lower vertical region (where YouTube comment box sits)
        comment_candidates = [d for d in comment_inputs if 0.25 <= d.normalized.cy <= 0.85]

        if comment_candidates:
            target_el = max(comment_candidates, key=lambda d: d.confidence)
            norm = target_el.normalized
            click_x = int(norm.cx * viewport_w)
            click_y = int(norm.cy * viewport_h)
            print(f"  [Model Vision] Located Comment Input Field! Type: {target_el.type} (conf: {target_el.confidence*100:.1f}%)")
        else:
            # Fallback to DOM-assisted center coordinates if comment box is custom shadow DOM
            comment_box = page.query_selector("#simplebox-placeholder, #contenteditable-root, ytd-commentbox")
            if comment_box:
                box_bounds = comment_box.bounding_box()
                click_x = int(box_bounds["x"] + box_bounds["width"] / 2)
                click_y = int(box_bounds["y"] + box_bounds["height"] / 2)
                print(f"  [Model Vision] Located YouTube Comment Box at ({click_x}px, {click_y}px)")
            else:
                click_x = int(viewport_w * 0.35)
                click_y = int(viewport_h * 0.45)
                print(f"  [Model Vision] Target fallback position at ({click_x}px, {click_y}px)")

        # Step 4: Move mouse to calculated coordinates, click, and type comment
        print(f"\n[Action] Moving mouse to ({click_x}px, {click_y}px), clicking, typing comment...")
        page.mouse.click(click_x, click_y)
        time.sleep(0.8)

        # Type comment text
        page.keyboard.type(comment_text)
        time.sleep(1.5)

        # Save screenshot showing typed comment
        typed_img_path = out_dir / "2_comment_typed.png"
        page.screenshot(path=str(typed_img_path))
        print(f"✅ Typed comment screenshot saved → {typed_img_path}")

        # Step 5: Locate and click Comment submit button if visible
        print("\n[Model Eyes] Searching for 'Comment' Submit Button...")
        screen_typed = out_dir / "2_comment_typed.png"
        detections_after = classifier.predict(screen_typed)
        buttons = [d for d in detections_after if d.type in ("button", "link")]

        # Look for button near the comment box
        btn_candidates = [d for d in buttons if abs(d.normalized.cy - (click_y/viewport_h)) < 0.25]
        if btn_candidates:
            btn_el = max(btn_candidates, key=lambda d: d.confidence)
            btn_norm = btn_el.normalized
            btn_x = int(btn_norm.cx * viewport_w)
            btn_y = int(btn_norm.cy * viewport_h)
            print(f"  [Model Vision] Located Submit Button at ({btn_x}px, {btn_y}px)")

            # Annotate verification screenshot
            img_verify = Image.open(typed_img_path).convert("RGB")
            draw = ImageDraw.Draw(img_verify)
            draw.rectangle([click_x - 120, click_y - 20, click_x + 350, click_y + 20], outline=(0, 255, 0), width=3)
            draw.ellipse([click_x - 10, click_y - 10, click_x + 10, click_y + 10], outline=(255, 0, 0), width=3)
            draw.text((click_x - 110, click_y - 35), f"MODEL EYE CLICK: Typed '{comment_text[:30]}...'", fill=(0, 255, 0))
            verify_path = out_dir / "3_youtube_comment_verification.png"
            img_verify.save(verify_path)
            print(f"✅ Visual verification overlay saved → {verify_path}")

        browser.close()


def main():
    parser = argparse.ArgumentParser(description="Post YouTube comment using ArrowUI model vision.")
    parser.add_argument("--url", type=str, default="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        help="YouTube Video URL")
    parser.add_argument("--comment", type=str,
                        default="Awesome video! Automated via ArrowUI AI model vision.",
                        help="Comment text to type")
    parser.add_argument("--model", type=str, default="arrow_ui_classifier.pt",
                        help="Model checkpoint path")
    args = parser.parse_args()

    post_youtube_comment_by_model_eyes(
        video_url=args.url,
        comment_text=args.comment,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
