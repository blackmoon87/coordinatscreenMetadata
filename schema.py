"""
schema.py — Pydantic data models for CartesianCoordinatesScreenMetadata dataset.
Defines the canonical JSON structure for every captured page and element.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 16 UI element classes  (index = class_id used in training labels)
# ---------------------------------------------------------------------------
CLASSES = [
    "button",          # 0
    "link",            # 1
    "input_text",      # 2
    "input_checkbox",  # 3
    "input_radio",     # 4
    "dropdown",        # 5
    "textarea",        # 6
    "image",           # 7
    "heading",         # 8
    "text",            # 9
    "list",            # 10
    "list_item",       # 11
    "navigation",      # 12
    "form",            # 13
    "table",           # 14
    "media",           # 15
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Map resolve_element_type() output → one of the 16 canonical classes
# Types not in CLASSES get remapped to the nearest parent class
TYPE_REMAP = {
    "input_email":     "input_text",
    "input_password":  "input_text",
    "input_search":    "input_text",
    "input_tel":       "input_text",
    "input_url":       "input_text",
    "input_date":      "input_text",
    "input_time":      "input_text",
    "input_number":    "input_text",
    "input_range":     "input_text",
    "input_color":     "input_text",
    "input_file":      "input_text",
    "input_hidden":    "input_text",
    "hidden":          "input_text",
    "table_header":    "table",
    "table_body":      "table",
    "table_row":       "table",
    "table_cell":      "table",
    "header":          "navigation",
    "footer":          "navigation",
    "sidebar":         "navigation",
    "main":            "container",
    "section":         "container",
    "article":         "container",
    "embed":           "image",
    "modal":           "container",
    "accordion":       "container",
    "code":            "text",
    "quote":           "text",
    "divider":         "text",
    "progress":        "text",
    "label":           "text",
    "container":       None,  # None = skip (not useful for classification)
}


def canonical_class(raw_type: str) -> Optional[str]:
    """Map any raw element type → one of 16 canonical class names, or None to skip."""
    if raw_type in CLASSES:
        return raw_type
    return TYPE_REMAP.get(raw_type, None)


# ---------------------------------------------------------------------------
# Semantic type mapping from raw HTML tags + input types
# ---------------------------------------------------------------------------
def resolve_element_type(tag: str, input_type: Optional[str] = None,
                          role: Optional[str] = None) -> str:
    """
    Maps raw HTML tag + optional input[type] attribute → clean semantic type string.
    This is what goes in the 'type' field of the output JSON.
    """
    tag = (tag or "").lower()
    input_type = (input_type or "").lower()
    role = (role or "").lower()

    # --- Input variants ---
    if tag == "input":
        mapping = {
            "text":     "input_text",
            "email":    "input_email",
            "password": "input_password",
            "number":   "input_number",
            "search":   "input_search",
            "tel":      "input_tel",
            "url":      "input_url",
            "date":     "input_date",
            "time":     "input_time",
            "checkbox": "input_checkbox",
            "radio":    "input_radio",
            "file":     "input_file",
            "range":    "input_range",
            "color":    "input_color",
            "submit":   "button",
            "button":   "button",
            "reset":    "button",
            "hidden":   "hidden",
            "image":    "button",
        }
        return mapping.get(input_type, "input_text")

    # --- Semantic HTML tags ---
    tag_map = {
        "button":   "button",
        "a":        "link",
        "img":      "image",
        "picture":  "image",
        "svg":      "image",
        "canvas":   "image",
        "video":    "media",
        "audio":    "media",
        "iframe":   "embed",
        "select":   "dropdown",
        "textarea": "textarea",
        "form":     "form",
        "table":    "table",
        "thead":    "table_header",
        "tbody":    "table_body",
        "tr":       "table_row",
        "td":       "table_cell",
        "th":       "table_cell",
        "ul":       "list",
        "ol":       "list",
        "li":       "list_item",
        "dl":       "list",
        "dt":       "list_item",
        "dd":       "list_item",
        "h1":       "heading",
        "h2":       "heading",
        "h3":       "heading",
        "h4":       "heading",
        "h5":       "heading",
        "h6":       "heading",
        "p":        "text",
        "span":     "text",
        "label":    "label",
        "nav":      "navigation",
        "header":   "header",
        "footer":   "footer",
        "main":     "main",
        "aside":    "sidebar",
        "section":  "section",
        "article":  "article",
        "dialog":   "modal",
        "details":  "accordion",
        "summary":  "accordion",
        "code":     "code",
        "pre":      "code",
        "blockquote": "quote",
        "hr":       "divider",
        "progress": "progress",
        "meter":    "progress",
        "figure":   "image",
        "figcaption": "text",
        "time":     "text",
        "abbr":     "text",
        "mark":     "text",
        "strong":   "text",
        "em":       "text",
        "div":      "container",
        "body":     "container",
        "section":  "container",
        "html":     "container",
    }

    # ARIA role overrides
    if role:
        role_map = {
            "button":     "button",
            "link":       "link",
            "textbox":    "input_text",
            "combobox":   "dropdown",
            "checkbox":   "input_checkbox",
            "radio":      "input_radio",
            "listbox":    "list",
            "option":     "list_item",
            "menu":       "list",
            "menuitem":   "list_item",
            "tab":        "button",
            "tablist":    "list",
            "dialog":     "modal",
            "alert":      "text",
            "img":        "image",
            "navigation": "navigation",
            "heading":    "heading",
            "search":     "form",
        }
        if role in role_map:
            return role_map[role]

    return tag_map.get(tag, "container")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CartesianBox(BaseModel):
    """Exact pixel bounding box in page-absolute coordinates (origin = top-left)."""
    x1: float = Field(..., description="Left edge (px)")
    y1: float = Field(..., description="Top edge (px)")
    x2: float = Field(..., description="Right edge (px)")
    y2: float = Field(..., description="Bottom edge (px)")
    cx: float = Field(..., description="Center X (px)")
    cy: float = Field(..., description="Center Y (px)")
    width: float = Field(..., description="Width (px)")
    height: float = Field(..., description="Height (px)")


class ElementStyle(BaseModel):
    """Relevant computed CSS properties for the element."""
    color: Optional[str] = None
    background_color: Optional[str] = None
    font_size: Optional[str] = None
    font_weight: Optional[str] = None
    display: Optional[str] = None
    visibility: Optional[str] = None
    opacity: Optional[str] = None
    z_index: Optional[str] = None
    border_radius: Optional[str] = None
    cursor: Optional[str] = None


class ElementAttributes(BaseModel):
    """Common HTML attributes captured for each element."""
    id: Optional[str] = None
    name: Optional[str] = None
    class_name: Optional[str] = None
    href: Optional[str] = None
    src: Optional[str] = None
    alt: Optional[str] = None
    placeholder: Optional[str] = None
    aria_label: Optional[str] = None
    aria_role: Optional[str] = None
    value: Optional[str] = None
    for_attr: Optional[str] = None
    data_attrs: dict = Field(default_factory=dict)


class ElementRecord(BaseModel):
    """
    Complete record for one screen element.
    This is the core unit of the dataset.
    """
    element_id: str = Field(..., description="Unique ID within this page session, e.g. 'el_0042'")
    tag: str = Field(..., description="Raw HTML tag name, e.g. 'button', 'input', 'img'")
    type: str = Field(..., description="Semantic element type (resolved from tag + attrs)")
    text: str = Field("", description="Visible text content of the element (trimmed)")
    cartesian: CartesianBox
    attributes: ElementAttributes
    style: ElementStyle
    is_visible: bool = Field(True, description="True if element is visible in the page")
    is_interactive: bool = Field(False, description="True if element is clickable/focusable")
    depth: int = Field(0, description="DOM tree depth (nesting level)")
    parent_id: Optional[str] = Field(None, description="element_id of nearest recorded parent")
    xpath: Optional[str] = Field(None, description="Absolute XPath to this element")


class ViewportInfo(BaseModel):
    width: int
    height: int


class PageDimensions(BaseModel):
    width: float
    height: float


class PageRecord(BaseModel):
    """Top-level record for one captured page."""
    session_id: str
    page_url: str
    page_title: str = ""
    captured_at: str
    viewport: ViewportInfo
    page_dimensions: PageDimensions
    screenshot_path: str = ""
    annotated_path: str = ""
    total_elements: int = 0
    interactive_elements: int = 0
    elements: list[ElementRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference output models (ArrowUI pipeline output)
# ---------------------------------------------------------------------------

class NormalizedBox(BaseModel):
    """
    Device-agnostic bounding box — all values in [0.0, 1.0].
    cx, cy = center point. w, h = width and height.
    Works for ANY screen size. Multiply by actual screenshot
    dimensions to get pixel coords on whatever device you have.
    """
    cx: float
    cy: float
    w: float
    h: float

    def to_pixels(self, img_width: int, img_height: int) -> dict:
        """Convert to absolute pixel coordinates for ANY screen size."""
        x1 = int((self.cx - self.w / 2) * img_width)
        y1 = int((self.cy - self.h / 2) * img_height)
        x2 = int((self.cx + self.w / 2) * img_width)
        y2 = int((self.cy + self.h / 2) * img_height)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1}


class DetectionResult(BaseModel):
    """
    One detected UI element — output of ArrowUI classifier.
    Mirrors ArrowClassifier.predict_with_ood() output contract.
    """
    element_id: str
    type: str                    # one of CLASSES
    class_id: int                # index in CLASSES
    text: str                    # visible text content
    confidence: float            # max softmax probability [0.0, 1.0]
    is_ood: bool                 # True = unknown / out-of-distribution
    normalized: NormalizedBox    # resolution-independent coords [0-1]
