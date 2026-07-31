"""
annotator.py — Draws color-coded bounding boxes on screenshots.

Takes a PageRecord (elements.json) + screenshot PNG and produces
an annotated PNG with color-coded boxes, type labels, and element IDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

from schema import PageRecord, ElementRecord

# ---------------------------------------------------------------------------
# Color palette per semantic element type
# ---------------------------------------------------------------------------
TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    # Interactive
    "button":           (50,  205, 50),    # lime green
    "link":             (30,  144, 255),   # dodger blue
    "input_text":       (255, 140, 0),     # dark orange
    "input_email":      (255, 140, 0),
    "input_password":   (255, 140, 0),
    "input_number":     (255, 140, 0),
    "input_search":     (255, 165, 0),
    "input_tel":        (255, 140, 0),
    "input_url":        (255, 140, 0),
    "input_date":       (255, 140, 0),
    "input_checkbox":   (218, 112, 214),   # orchid
    "input_radio":      (218, 112, 214),
    "input_file":       (255, 99,  71),    # tomato
    "input_range":      (255, 165, 0),
    "input_color":      (255, 165, 0),
    "dropdown":         (255, 215, 0),     # gold
    "textarea":         (255, 140, 0),
    # Content
    "image":            (147, 112, 219),   # medium purple
    "media":            (100, 149, 237),   # cornflower blue
    "text":             (169, 169, 169),   # dark gray
    "heading":          (255, 69,  0),     # red-orange
    "label":            (176, 196, 222),   # light steel blue
    "list":             (95,  158, 160),   # cadet blue
    "list_item":        (95,  158, 160),
    "table":            (60,  179, 113),   # medium sea green
    "table_row":        (60,  179, 113),
    "table_cell":       (60,  179, 113),
    # Layout / Structure
    "form":             (255, 99,  71),    # tomato
    "navigation":       (64,  224, 208),   # turquoise
    "header":           (70,  130, 180),   # steel blue
    "footer":           (70,  130, 180),
    "section":          (119, 136, 153),   # light slate gray
    "article":          (119, 136, 153),
    "modal":            (220, 20,  60),    # crimson
    "container":        (211, 211, 211),   # light gray
    "embed":            (100, 149, 237),
    "code":             (255, 215, 0),
    "quote":            (169, 169, 169),
    "divider":          (169, 169, 169),
    "progress":         (50,  205, 50),
}

DEFAULT_COLOR = (200, 200, 200)  # fallback gray


def _get_color(element_type: str, alpha: int = 180) -> tuple:
    rgb = TYPE_COLORS.get(element_type, DEFAULT_COLOR)
    return rgb + (alpha,)


def annotate(
    page_record: PageRecord,
    output_path: Union[str, Path],
    min_width: float = 5.0,
    min_height: float = 5.0,
    show_labels: bool = True,
    label_font_size: int = 11,
) -> Path:
    """
    Draw color-coded bounding boxes on the screenshot in `page_record`.

    Args:
        page_record:     PageRecord containing elements and screenshot_path.
        output_path:     Where to save the annotated PNG.
        min_width:       Skip boxes narrower than this (px).
        min_height:      Skip boxes shorter than this (px).
        show_labels:     Whether to draw type + text labels.
        label_font_size: Font size for labels.

    Returns:
        Path to the saved annotated image.
    """
    screenshot = Path(page_record.screenshot_path)
    if not screenshot.exists():
        raise FileNotFoundError(f"Screenshot not found: {screenshot}")

    base_img = Image.open(screenshot).convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a system font; fall back to PIL default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", label_font_size)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(8, label_font_size - 2))
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Sort so containers render first (behind interactive elements)
    def sort_key(e: ElementRecord) -> int:
        priority = {"container": 0, "section": 1, "article": 1, "form": 2}
        return priority.get(e.type, 5)

    sorted_elements = sorted(page_record.elements, key=sort_key)

    for el in sorted_elements:
        b = el.cartesian
        if b.width < min_width or b.height < min_height:
            continue

        # Clamp coordinates to image bounds
        img_w, img_h = base_img.size
        x1 = max(0, int(b.x1))
        y1 = max(0, int(b.y1))
        x2 = min(img_w - 1, int(b.x2))
        y2 = min(img_h - 1, int(b.y2))

        if x2 <= x1 or y2 <= y1:
            continue

        color_fill = _get_color(el.type, alpha=25)
        color_border = _get_color(el.type, alpha=200)

        # Draw fill
        draw.rectangle([x1, y1, x2, y2], fill=color_fill)
        # Draw border (2px thick)
        for offset in range(2):
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=color_border,
            )

        # Label: "{type} | {text[:30]}"
        if show_labels and b.height >= 14:
            text_preview = el.text[:28] + "…" if len(el.text) > 28 else el.text
            label = f"{el.type}"
            if text_preview:
                label += '  "' + text_preview + '"'

            # Draw label background
            label_x = x1 + 3
            label_y = y1 + 2
            bbox = draw.textbbox((label_x, label_y), label, font=font_small)
            bg_x2 = min(bbox[2] + 2, img_w - 1)
            bg_y2 = min(bbox[3] + 1, img_h - 1)
            draw.rectangle(
                [label_x - 1, label_y - 1, bg_x2, bg_y2],
                fill=(0, 0, 0, 140),
            )
            draw.text((label_x, label_y), label, fill=(255, 255, 255, 255), font=font_small)

    # Composite overlay onto base image
    combined = Image.alpha_composite(base_img, overlay).convert("RGB")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(output_path), format="PNG")
    print(f"[annotator] Annotated image saved: {output_path}")
    return output_path


def build_legend(output_path: Union[str, Path]) -> Path:
    """
    Generates a standalone PNG legend showing all element types and their colors.
    """
    types = list(TYPE_COLORS.keys())
    cell_h = 24
    cell_w = 280
    cols = 3
    rows = (len(types) + cols - 1) // cols
    pad = 10

    img = Image.new("RGB", (cell_w * cols + pad * 2, cell_h * rows + pad * 2 + 30), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    draw.text((pad, pad), "Element Type Color Legend", fill=(255, 255, 255), font=title_font)

    for i, etype in enumerate(types):
        col = i % cols
        row = i // cols
        x = pad + col * cell_w
        y = pad + 30 + row * cell_h
        rgb = TYPE_COLORS[etype]
        draw.rectangle([x, y + 3, x + 18, y + cell_h - 3], fill=rgb)
        draw.text((x + 24, y + 4), etype, fill=(220, 220, 220), font=font)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    print(f"[annotator] Legend saved: {output_path}")
    return output_path
