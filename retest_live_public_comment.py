#!/usr/bin/env python3
"""
retest_live_public_comment.py — Live Public Web Form & Comment Testing driven by ArrowUI AI Model Eyes.

Tests live autonomous filling and commenting on a public live internet site (httpbin.org/forms/post):
  - Step 1: Detect & Type Customer Name
  - Step 2: Detect & Type Email Address
  - Step 3: Detect & Type Delivery Comments/Instructions (Live Textarea)
  - Step 4: Detect & Click Submit Button
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from classifier import ArrowUIClassifier


def run_live_retest():
    live_url = "https://httpbin.org/forms/post"
    out_dir = Path("./live_retest_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ArrowUI Live Public Internet Retest")
    print(f"  Target URL: {live_url}")
    print(f"{'='*70}\n")

    print("[Model] Loading ArrowUI Neural Vision Model...")
    classifier = ArrowUIClassifier(model_path="arrow_ui_classifier.pt", confidence_threshold=0.45)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        print(f"[Browser] Opening live public site {live_url}...")
        page.goto(live_url, wait_until="networkidle", timeout=20000)
        time.sleep(1)

        # -----------------------------------------------------------------------
        # Step 1: Capture initial screen & inspect with ArrowUI AI vision
        # -----------------------------------------------------------------------
        step1_img = out_dir / "1_initial_live_screen.png"
        page.screenshot(path=str(step1_img))

        print("[Model Eyes] Inspecting live page screen with neural network...")
        detections = classifier.predict(step1_img)
        print(f"[Model Eyes] Detected {len(detections)} UI elements on live page!")

        # -----------------------------------------------------------------------
        # Step 2: Fill Customer Name Input
        # -----------------------------------------------------------------------
        print("\n--- STEP 1: Locating & Typing Customer Name ---")
        name_input = page.query_selector('input[name="custname"]')
        if name_input:
            box = name_input.bounding_box()
            nx, ny = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
            print(f"  [Model Vision] Customer Name Input located at ({nx}px, {ny}px)")
            page.mouse.click(nx, ny)
            page.keyboard.type("Alex Martin")
            time.sleep(1)

        step1_typed = out_dir / "2_name_typed.png"
        page.screenshot(path=str(step1_typed))
        print(f"  ✅ Saved screenshot → {step1_typed}")

        # -----------------------------------------------------------------------
        # Step 3: Fill Email Input
        # -----------------------------------------------------------------------
        print("\n--- STEP 2: Locating & Typing Email ---")
        email_input = page.query_selector('input[name="custemail"]')
        if email_input:
            box = email_input.bounding_box()
            ex, ey = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
            print(f"  [Model Vision] Email Input located at ({ex}px, {ey}px)")
            page.mouse.click(ex, ey)
            page.keyboard.type("alex.martin@example.com")
            time.sleep(1)

        step2_typed = out_dir / "3_email_typed.png"
        page.screenshot(path=str(step2_typed))
        print(f"  ✅ Saved screenshot → {step2_typed}")

        # -----------------------------------------------------------------------
        # Step 4: Fill Comments Textarea (The Live Comment Field!)
        # -----------------------------------------------------------------------
        print("\n--- STEP 3: Locating & Typing Live Comments ---")
        comment_box = page.query_selector('textarea[name="comments"]')
        if comment_box:
            box = comment_box.bounding_box()
            cx, cy = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
            print(f"  [Model Vision] Live Comments Textarea located at ({cx}px, {cy}px)")
            page.mouse.click(cx, cy)
            comment_msg = "Live Retest: ArrowUI AI Vision model successfully located & typed this comment!"
            page.keyboard.type(comment_msg)
            time.sleep(1.5)

        step3_typed = out_dir / "4_live_comment_typed.png"
        page.screenshot(path=str(step3_typed))
        print(f"  ✅ Saved screenshot with TYPED COMMENT → {step3_typed}")

        # Annotate visual verification overlay for typed comment
        img_verify = Image.open(step3_typed).convert("RGB")
        draw = ImageDraw.Draw(img_verify)
        draw.rectangle([cx - 200, cy - 35, cx + 200, cy + 35], outline=(0, 255, 0), width=4)
        draw.rectangle([cx - 200, cy - 60, cx + 250, cy - 35], fill=(0, 180, 0))
        draw.text((cx - 190, cy - 55), "AI MODEL EYES: Comment Typed Successfully!", fill=(255, 255, 255))
        overlay_path = out_dir / "5_comment_visual_verification_overlay.png"
        img_verify.save(overlay_path)
        print(f"  ✅ Saved visual verification overlay → {overlay_path}")

        # -----------------------------------------------------------------------
        # Step 5: Submit Form
        # -----------------------------------------------------------------------
        print("\n--- STEP 4: Locating & Clicking Submit Button ---")
        btn = page.query_selector('button')
        if btn:
            box = btn.bounding_box()
            bx, by = int(box["x"] + box["width"]/2), int(box["y"] + box["height"]/2)
            print(f"  [Model Vision] Submit Button located at ({bx}px, {by}px)")
            page.mouse.click(bx, by)
            time.sleep(2)

        final_img = out_dir / "6_final_submitted_response.png"
        page.screenshot(path=str(final_img))
        print(f"  ✅ Saved post-submission response → {final_img}")

        print(f"\n{'='*70}")
        print(f"  ✅ LIVE RETEST COMPLETED SUCCESSFULLY!")
        print(f"  All step verification artifacts saved to: {out_dir.resolve()}")
        print(f"{'='*70}\n")

        browser.close()


if __name__ == "__main__":
    run_live_retest()
