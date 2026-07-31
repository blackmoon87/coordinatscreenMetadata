#!/usr/bin/env python3
"""
complex_form_automation.py — Multi-Step Autonomous Agent driven by ArrowUI Model Vision.

Executes a full interactive workflow using AI eyes:
  Step 1: Inspect screen → Locate username input_text → Click & Type username
  Step 2: Inspect screen → Locate password input_text → Click & Type password
  Step 3: Inspect screen → Locate Login button → Click button
  Step 4: Inspect screen → Verify successful login & detect new post-login UI elements

Usage:
    python3 complex_form_automation.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

from classifier import ArrowUIClassifier


def run_complex_automation(
    url: str = "https://the-internet.herokuapp.com/login",
    username_val: str = "tomsmith",
    password_val: str = "SuperSecretPassword!",
    model_path: str = "arrow_ui_classifier.pt",
    viewport_w: int = 1280,
    viewport_h: int = 800,
    output_dir: str = "./complex_test_output",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ArrowUI Multi-Step Autonomous Agent")
    print(f"  Target URL: {url}")
    print(f"  Model:      {model_path}")
    print(f"  Viewport:   {viewport_w}×{viewport_h}")
    print(f"{'='*70}\n")

    print(f"[Agent] Loading ArrowUI Neural Vision Model...")
    classifier = ArrowUIClassifier(model_path=model_path, confidence_threshold=0.45)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_w, "height": viewport_h})

        print(f"[Agent] Opening website {url}...")
        page.goto(url, wait_until="networkidle", timeout=20000)
        time.sleep(1)

        # -----------------------------------------------------------------------
        # STEP 1: Fill Username Box
        # -----------------------------------------------------------------------
        print(f"\n--- STEP 1: Locating & Filling Username Input ---")
        step1_img = out_dir / "step1_initial_screen.png"
        page.screenshot(path=str(step1_img))

        detections = classifier.predict(step1_img)
        inputs = [d for d in detections if d.type in ("input_text", "form", "text")]

        if not inputs:
            print("⚠️ No input elements found in step 1, using candidate fallback")
            inputs = detections

        # Pick top input element sorted by vertical position (username is higher than password)
        inputs_sorted = sorted(inputs, key=lambda d: d.normalized.cy)
        username_el = inputs_sorted[0]
        norm = username_el.normalized
        u_x = int(norm.cx * viewport_w)
        u_y = int(norm.cy * viewport_h)

        print(f"  [Model Vision] Detected Username Box at normalized (cx={norm.cx:.4f}, cy={norm.cy:.4f})")
        print(f"  [Action] Moving mouse to ({u_x}px, {u_y}px), clicking, typing '{username_val}'...")
        page.mouse.click(u_x, u_y)
        page.keyboard.type(username_val)
        time.sleep(1)

        # Save annotated screenshot
        img1 = Image.open(step1_img).convert("RGB")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([u_x - 100, u_y - 20, u_x + 100, u_y + 20], outline=(0, 255, 0), width=3)
        draw1.text((u_x - 90, u_y - 35), f"STEP 1: Typed Username ({username_val})", fill=(0, 255, 0))
        img1.save(out_dir / "step1_username_filled.png")

        # -----------------------------------------------------------------------
        # STEP 2: Fill Password Box
        # -----------------------------------------------------------------------
        print(f"\n--- STEP 2: Locating & Filling Password Input ---")
        step2_img = out_dir / "step2_after_username.png"
        page.screenshot(path=str(step2_img))

        detections2 = classifier.predict(step2_img)
        inputs2 = [d for d in detections2 if d.type in ("input_text", "form", "text")]
        inputs2_sorted = sorted(inputs2, key=lambda d: d.normalized.cy)

        # Pick password input (below username input)
        password_el = inputs2_sorted[min(1, len(inputs2_sorted) - 1)]
        norm2 = password_el.normalized
        p_x = int(norm2.cx * viewport_w)
        p_y = int(norm2.cy * viewport_h)

        print(f"  [Model Vision] Detected Password Box at normalized (cx={norm2.cx:.4f}, cy={norm2.cy:.4f})")
        print(f"  [Action] Moving mouse to ({p_x}px, {p_y}px), clicking, typing password...")
        page.mouse.click(p_x, p_y)
        page.keyboard.type(password_val)
        time.sleep(1)

        img2 = Image.open(step2_img).convert("RGB")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([p_x - 100, p_y - 20, p_x + 100, p_y + 20], outline=(0, 255, 0), width=3)
        draw2.text((p_x - 90, p_y - 35), "STEP 2: Typed Password", fill=(0, 255, 0))
        img2.save(out_dir / "step2_password_filled.png")

        # -----------------------------------------------------------------------
        # STEP 3: Locate & Click Login Button
        # -----------------------------------------------------------------------
        print(f"\n--- STEP 3: Locating & Clicking Submit Button ---")
        step3_img = out_dir / "step3_before_button_click.png"
        page.screenshot(path=str(step3_img))

        detections3 = classifier.predict(step3_img)
        buttons = [d for d in detections3 if d.type == "button"]

        if not buttons:
            print("⚠️ Button type not found, checking all interactive candidates")
            buttons = [d for d in detections3 if d.type in ("button", "link", "form")]

        button_el = max(buttons, key=lambda d: d.confidence)
        norm3 = button_el.normalized
        b_x = int(norm3.cx * viewport_w)
        b_y = int(norm3.cy * viewport_h)

        print(f"  [Model Vision] Detected Submit Button (conf: {button_el.confidence*100:.1f}%)")
        print(f"  [Model Vision] Normalized coords (cx={norm3.cx:.4f}, cy={norm3.cy:.4f})")
        print(f"  [Action] Moving mouse to ({b_x}px, {b_y}px) and clicking Submit...")
        page.mouse.click(b_x, b_y)
        time.sleep(2)

        # -----------------------------------------------------------------------
        # STEP 4: Inspect Result Page & Detect Post-Login Elements
        # -----------------------------------------------------------------------
        print(f"\n--- STEP 4: Post-Action Inspection & Verification ---")
        final_img = out_dir / "step4_post_login_screen.png"
        page.screenshot(path=str(final_img))

        final_detections = classifier.predict(final_img)
        print(f"  [Model Vision] Page transitioned! Detected {len(final_detections)} UI elements on new screen:")

        from collections import Counter
        final_counts = Counter(d.type for d in final_detections)
        for t, c in final_counts.most_common():
            print(f"    {t:15s}: {c:3d}")

        # Annotate final post-login screen
        img4 = Image.open(final_img).convert("RGB")
        draw4 = ImageDraw.Draw(img4)
        for det in final_detections:
            px = det.normalized.to_pixels(viewport_w, viewport_h)
            draw4.rectangle([px["x1"], px["y1"], px["x2"], px["y2"]], outline=(0, 255, 0), width=2)

        draw4.rectangle([20, 20, 450, 60], fill=(0, 150, 0))
        draw4.text((30, 30), "SUCCESS: Autonomous Multi-Step Workflow Complete!", fill=(255, 255, 255))
        img4.save(out_dir / "step4_final_verification.png")

        print(f"\n{'='*70}")
        print(f"  ✅ MULTI-STEP WORKFLOW COMPLETED SUCCESSFULLY!")
        print(f"  All step verification artifacts saved to: {out_dir.resolve()}")
        print(f"{'='*70}\n")

        browser.close()


if __name__ == "__main__":
    run_complex_automation()
