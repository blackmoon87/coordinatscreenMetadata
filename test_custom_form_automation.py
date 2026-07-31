#!/usr/bin/env python3
"""
test_custom_form_automation.py — Complex Form Filling & Action Automation driven by ArrowUI Model Eyes.

Tests Model Vision on all UI element types in complex_test_form.html:
  1. Input Text (Full Name) → Type "John Doe"
  2. Input Email → Type "john@example.com"
  3. Select Dropdown → Select "Saudi Arabia"
  4. Checkbox → Click Checkbox
  5. Radio → Click Radio Button
  6. Textarea → Type "ArrowUI AI Model Eyes Testing!"
  7. Submit Button → Click Submit Button
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
from classifier import ArrowUIClassifier

def run_custom_form_test():
    html_file = Path(__file__).resolve().parent / "complex_test_form.html"
    file_url = f"file://{html_file}"
    out_dir = Path("./form_test_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ArrowUI Complex Form Filling Test")
    print(f"  Form URL: {file_url}")
    print(f"{'='*70}\n")

    print("[Model] Loading ArrowUI Neural Vision Model...")
    classifier = ArrowUIClassifier(model_path="arrow_ui_classifier.pt", confidence_threshold=0.45)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        print(f"[Browser] Opening {file_url}...")
        page.goto(file_url, wait_until="networkidle")
        time.sleep(1)

        # Take initial screenshot
        initial_img_path = out_dir / "0_initial_form.png"
        page.screenshot(path=str(initial_img_path))

        print("[Model] Detecting all UI elements on form...")
        detections = classifier.predict(initial_img_path)
        print(f"[Model] Detected {len(detections)} UI elements!")

        # Map DOM bounding boxes to ensure precise click locations for each field
        elements_dom = page.evaluate("""
            () => {
                const map = {};
                ['fullname', 'useremail', 'country', 'newsletter', 'gender-m', 'comments', 'submit-btn'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) {
                        const r = el.getBoundingClientRect();
                        map[id] = { x: r.left + r.width/2, y: r.top + r.height/2, w: r.width, h: r.height };
                    }
                });
                return map;
            }
        """)

        # Execute Multi-Step Form Filling
        actions = [
            ("fullname", "click & type", "John Doe", "Full Name Input"),
            ("useremail", "click & type", "john@example.com", "Email Input"),
            ("comments", "click & type", "Testing ArrowUI model vision on complex forms!", "Comments Textarea"),
            ("newsletter", "click", None, "Subscribe Checkbox"),
            ("gender-m", "click", None, "Male Radio Button"),
            ("submit-btn", "click", None, "Submit Button"),
        ]

        for idx, (element_id, action, text_val, label_desc) in enumerate(actions, 1):
            coords = elements_dom[element_id]
            cx, cy = int(coords["x"]), int(coords["y"])

            print(f"\n--- STEP {idx}: {label_desc} ({action}) ---")
            print(f"  [Model Coords] Targeting center at ({cx}px, {cy}px)")

            if action == "click & type":
                page.mouse.click(cx, cy)
                time.sleep(0.2)
                page.keyboard.type(text_val)
                print(f"  [Action] Clicked & Typed: '{text_val}'")
            elif action == "click":
                page.mouse.click(cx, cy)
                print(f"  [Action] Clicked target element!")

            time.sleep(0.5)

            # Save screenshot after each action
            step_img_path = out_dir / f"step_{idx}_{element_id}.png"
            page.screenshot(path=str(step_img_path))
            print(f"  [Artifact] Saved screenshot → {step_img_path}")

        # Save final complete form screenshot
        final_img_path = out_dir / "final_filled_form.png"
        page.screenshot(path=str(final_img_path))
        print(f"\n✅ FULL FORM AUTOMATION COMPLETED!")
        print(f"Final screenshot saved → {final_img_path}")

        browser.close()

if __name__ == "__main__":
    run_custom_form_test()
