"""
extractor.py — Playwright-based page element extractor.

Visits a URL (or local HTML file), injects JavaScript to traverse the DOM,
calls getBoundingClientRect() on every element, captures computed styles,
and returns structured ElementRecord objects with exact Cartesian coordinates.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Page

from schema import (
    CartesianBox,
    ElementAttributes,
    ElementRecord,
    ElementStyle,
    PageDimensions,
    PageRecord,
    ViewportInfo,
    resolve_element_type,
)

# ---------------------------------------------------------------------------
# JavaScript injected into the page to extract all element data
# ---------------------------------------------------------------------------
_JS_EXTRACT = """
() => {
    const INTERACTIVE_TAGS = new Set(['a', 'button', 'input', 'select', 'textarea', 'label']);
    const INTERACTIVE_ROLES = new Set(['button','link','checkbox','radio','textbox',
                                        'combobox','listbox','menuitem','tab','switch','option']);
    const SKIP_TAGS = new Set(['script','style','meta','head','noscript','template','br','wbr']);

    function getVisibleText(el) {
        // Get direct visible text only (not deep children text to avoid duplication)
        let text = '';
        for (const node of el.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                text += node.textContent;
            }
        }
        // For inputs, use value or placeholder; for images, use alt
        const tag = el.tagName.toLowerCase();
        if ((tag === 'input' || tag === 'textarea') && !text.trim()) {
            text = el.value || el.placeholder || el.getAttribute('aria-label') || '';
        }
        if (tag === 'img') {
            text = el.alt || el.getAttribute('aria-label') || el.src || '';
        }
        return text.trim().replace(/\\s+/g, ' ').substring(0, 500);
    }

    function getDataAttrs(el) {
        const data = {};
        for (const attr of el.attributes) {
            if (attr.name.startsWith('data-')) {
                data[attr.name] = attr.value;
            }
        }
        return data;
    }

    function isInteractive(el) {
        const tag = el.tagName.toLowerCase();
        if (INTERACTIVE_TAGS.has(tag)) return true;
        const role = el.getAttribute('role') || '';
        if (INTERACTIVE_ROLES.has(role.toLowerCase())) return true;
        const tabindex = el.getAttribute('tabindex');
        if (tabindex !== null && parseInt(tabindex) >= 0) return true;
        const style = window.getComputedStyle(el);
        if (style.cursor === 'pointer') return true;
        return false;
    }

    function getDepth(el) {
        let depth = 0;
        let node = el;
        while (node.parentElement) { depth++; node = node.parentElement; }
        return depth;
    }

    function getXPath(el) {
        const parts = [];
        let node = el;
        while (node && node.nodeType === Node.ELEMENT_NODE) {
            let idx = 1;
            let sibling = node.previousSibling;
            while (sibling) {
                if (sibling.nodeType === Node.ELEMENT_NODE &&
                    sibling.tagName === node.tagName) idx++;
                sibling = sibling.previousSibling;
            }
            parts.unshift(`${node.tagName.toLowerCase()}[${idx}]`);
            node = node.parentElement;
        }
        return '/' + parts.join('/');
    }

    const scrollY = window.scrollY || 0;
    const scrollX = window.scrollX || 0;
    const allElements = document.querySelectorAll('*');
    const results = [];
    let idx = 0;

    for (const el of allElements) {
        const tag = el.tagName.toLowerCase();
        if (SKIP_TAGS.has(tag)) continue;

        const rect = el.getBoundingClientRect();
        // Skip zero-size or completely off-screen elements
        if (rect.width < 1 || rect.height < 1) continue;

        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') continue;
        if (parseFloat(style.opacity) < 0.01) continue;

        // Page-absolute coordinates (add scroll offset)
        const x1 = rect.left + scrollX;
        const y1 = rect.top + scrollY;
        const x2 = rect.right + scrollX;
        const y2 = rect.bottom + scrollY;

        results.push({
            element_id: 'el_' + String(idx).padStart(4, '0'),
            tag: tag,
            input_type: el.getAttribute('type') || '',
            aria_role: el.getAttribute('role') || '',
            text: getVisibleText(el),
            x1: Math.round(x1 * 10) / 10,
            y1: Math.round(y1 * 10) / 10,
            x2: Math.round(x2 * 10) / 10,
            y2: Math.round(y2 * 10) / 10,
            width: Math.round(rect.width * 10) / 10,
            height: Math.round(rect.height * 10) / 10,
            cx: Math.round((x1 + x2) / 2 * 10) / 10,
            cy: Math.round((y1 + y2) / 2 * 10) / 10,
            attr_id: el.id || '',
            attr_name: el.getAttribute('name') || '',
            attr_class: el.className || '',
            attr_href: el.getAttribute('href') || '',
            attr_src: el.getAttribute('src') || '',
            attr_alt: el.getAttribute('alt') || '',
            attr_placeholder: el.getAttribute('placeholder') || '',
            attr_aria_label: el.getAttribute('aria-label') || '',
            attr_value: el.value || '',
            attr_for: el.getAttribute('for') || '',
            data_attrs: getDataAttrs(el),
            style_color: style.color || '',
            style_background: style.backgroundColor || '',
            style_font_size: style.fontSize || '',
            style_font_weight: style.fontWeight || '',
            style_display: style.display || '',
            style_visibility: style.visibility || '',
            style_opacity: style.opacity || '',
            style_z_index: style.zIndex || '',
            style_border_radius: style.borderRadius || '',
            style_cursor: style.cursor || '',
            is_interactive: isInteractive(el),
            depth: getDepth(el),
            xpath: getXPath(el),
        });
        idx++;
    }

    return {
        elements: results,
        page_width: Math.max(document.body.scrollWidth, document.documentElement.scrollWidth),
        page_height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
        title: document.title,
    };
}
"""


# ---------------------------------------------------------------------------
# Core extractor function
# ---------------------------------------------------------------------------

def extract_page(
    url: str,
    screenshot_path: Optional[Path] = None,
    viewport_width: int = 1920,
    viewport_height: int = 1080,
    wait_for: str = "networkidle",
    timeout_ms: int = 30_000,
) -> PageRecord:
    """
    Visit `url` with Playwright, extract ALL visible elements with exact
    Cartesian coordinates and metadata.

    Args:
        url:              Full URL or file:// path to local HTML.
        screenshot_path:  If provided, saves a full-page PNG screenshot here.
        viewport_width:   Browser viewport width in pixels.
        viewport_height:  Browser viewport height in pixels.
        wait_for:         Playwright wait condition ('load', 'networkidle', 'domcontentloaded').
        timeout_ms:       Navigation timeout in milliseconds.

    Returns:
        PageRecord with all extracted elements.
    """
    session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    captured_at = datetime.now(timezone.utc).isoformat()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport_width, "height": viewport_height},
            device_scale_factor=1,
        )
        page: Page = context.new_page()

        print(f"[extractor] Loading: {url}")
        page.goto(url, wait_until=wait_for, timeout=timeout_ms)

        # Scroll to bottom to trigger lazy-loaded elements, then back to top
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)

        # Capture screenshot (full page)
        if screenshot_path:
            screenshot_path = Path(screenshot_path)
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"[extractor] Screenshot saved: {screenshot_path}")

        # Inject JS and extract element data
        raw = page.evaluate(_JS_EXTRACT)
        browser.close()

    # --- Parse raw JS output into typed models ---
    elements: list[ElementRecord] = []
    for r in raw["elements"]:
        sem_type = resolve_element_type(
            tag=r["tag"],
            input_type=r.get("input_type"),
            role=r.get("aria_role"),
        )

        cartesian = CartesianBox(
            x1=r["x1"], y1=r["y1"],
            x2=r["x2"], y2=r["y2"],
            cx=r["cx"], cy=r["cy"],
            width=r["width"], height=r["height"],
        )

        attrs = ElementAttributes(
            id=r["attr_id"] or None,
            name=r["attr_name"] or None,
            class_name=r["attr_class"] or None,
            href=r["attr_href"] or None,
            src=r["attr_src"] or None,
            alt=r["attr_alt"] or None,
            placeholder=r["attr_placeholder"] or None,
            aria_label=r["attr_aria_label"] or None,
            aria_role=r["aria_role"] or None,
            value=r["attr_value"] or None,
            for_attr=r["attr_for"] or None,
            data_attrs=r.get("data_attrs", {}),
        )

        style = ElementStyle(
            color=r["style_color"] or None,
            background_color=r["style_background"] or None,
            font_size=r["style_font_size"] or None,
            font_weight=r["style_font_weight"] or None,
            display=r["style_display"] or None,
            visibility=r["style_visibility"] or None,
            opacity=r["style_opacity"] or None,
            z_index=r["style_z_index"] or None,
            border_radius=r["style_border_radius"] or None,
            cursor=r["style_cursor"] or None,
        )

        elements.append(ElementRecord(
            element_id=r["element_id"],
            tag=r["tag"],
            type=sem_type,
            text=r["text"],
            cartesian=cartesian,
            attributes=attrs,
            style=style,
            is_visible=True,
            is_interactive=r["is_interactive"],
            depth=r["depth"],
            xpath=r.get("xpath"),
        ))

    interactive_count = sum(1 for e in elements if e.is_interactive)
    print(f"[extractor] Extracted {len(elements)} elements ({interactive_count} interactive)")

    return PageRecord(
        session_id=session_id,
        page_url=url,
        page_title=raw.get("title", ""),
        captured_at=captured_at,
        viewport=ViewportInfo(width=viewport_width, height=viewport_height),
        page_dimensions=PageDimensions(
            width=raw.get("page_width", viewport_width),
            height=raw.get("page_height", viewport_height),
        ),
        screenshot_path=str(screenshot_path) if screenshot_path else "",
        total_elements=len(elements),
        interactive_elements=interactive_count,
        elements=elements,
    )
