"""Parser utility to extract and standardize layout bounding boxes from Chandra OCR outputs.

Supports structured JSON blocks, HTML tags with data-bbox attributes,
and converts coordinates to DocLayNet standard [x, y, w, h] format.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

DOCLAYNET_CATEGORIES = {
    "Caption": 1,
    "Footnote": 2,
    "Formula": 3,
    "List-item": 4,
    "Page-footer": 5,
    "Page-header": 6,
    "Picture": 7,
    "Section-header": 8,
    "Table": 9,
    "Text": 10,
    "Title": 11,
}

# Mapping Chandra block types to DocLayNet 11 categories
CHANDRA_NORM_MAP = {
    # Text
    "text": "Text",
    "paragraph": "Text",
    "body": "Text",
    "plain_text": "Text",
    "code-block": "Text",
    "code_block": "Text",
    "bibliography": "Text",
    # Title & Headers
    "title": "Title",
    "document-title": "Title",
    "doc_title": "Title",
    "section-header": "Section-header",
    "section_header": "Section-header",
    "sub-header": "Section-header",
    "sub_header": "Section-header",
    "heading": "Section-header",
    "table-of-contents": "Section-header",
    "toc": "Section-header",
    # Running header & footer
    "page-header": "Page-header",
    "page_header": "Page-header",
    "header": "Page-header",
    "page-footer": "Page-footer",
    "page_footer": "Page-footer",
    "footer": "Page-footer",
    # Captions & Footnotes
    "caption": "Caption",
    "footnote": "Footnote",
    # Tables
    "table": "Table",
    "tabular": "Table",
    "form": "Table",
    # Images & Diagrams
    "image": "Picture",
    "figure": "Picture",
    "diagram": "Picture",
    "chart": "Picture",
    "picture": "Picture",
    "illustration": "Picture",
    "chemical-block": "Picture",
    # Math & Formulas
    "equation-block": "Formula",
    "equation_block": "Formula",
    "equation": "Formula",
    "formula": "Formula",
    "math": "Formula",
    # Lists
    "list-group": "List-item",
    "list_group": "List-item",
    "list-item": "List-item",
    "list_item": "List-item",
    "list": "List-item",
}


def parse_bbox_values(val: Any) -> Optional[List[float]]:
    """Converts diverse bbox representations (list, tuple, string) into [x1, y1, x2, y2]."""
    if isinstance(val, (list, tuple)) and len(val) >= 4:
        try:
            return [float(v) for v in val[:4]]
        except (ValueError, TypeError):
            return None
    elif isinstance(val, str):
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val)
        if len(nums) >= 4:
            try:
                return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
            except (ValueError, TypeError):
                return None
    return None


def parse_html_bboxes(raw_text: str) -> List[Dict[str, Any]]:
    """Extracts elements marked with data-bbox or class attributes from HTML output."""
    items = []
    # Match patterns like <tag ... data-bbox="10, 20, 100, 200" ... data-type="Section-header" ...>
    tag_matches = re.findall(
        r'<[^>]*?(?:data-bbox|bbox)=["\']([^"\']+)["\'][^>]*?>',
        raw_text,
        re.IGNORECASE,
    )
    if tag_matches:
        for bbox_str in tag_matches:
            coords = parse_bbox_values(bbox_str)
            if coords:
                items.append({"bbox": coords, "type": "Text"})

    # More specific search: class and data-bbox
    block_matches = re.findall(
        r'<([a-zA-Z0-9]+)\s+[^>]*?class=["\']([^"\']+)["\'][^>]*?data-bbox=["\']([^"\']+)["\'][^>]*?>',
        raw_text,
        re.IGNORECASE,
    )
    if block_matches:
        for _, class_str, bbox_str in block_matches:
            coords = parse_bbox_values(bbox_str)
            if coords:
                # Find matching category from class
                matched_cat = "Text"
                for cls in class_str.split():
                    cls_clean = cls.strip().lower()
                    if cls_clean in CHANDRA_NORM_MAP:
                        matched_cat = CHANDRA_NORM_MAP[cls_clean]
                        break
                items.append({"bbox": coords, "type": matched_cat})
    return items


def parse_chandra_layout_response(
    raw_response: str,
    img_width: int,
    img_height: int,
    default_score: float = 0.95,
) -> List[Dict[str, Any]]:
    """Parses raw text output from Chandra model into standard detection schema."""
    text = raw_response.strip()

    # 1. Try JSON block extraction
    data = None
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    candidate_json = json_match.group(1).strip() if json_match else text

    # Try brackets
    start_bracket = candidate_json.find("[")
    end_bracket = candidate_json.rfind("]")
    if start_bracket != -1 and end_bracket > start_bracket:
        try:
            data = json.loads(candidate_json[start_bracket : end_bracket + 1])
        except Exception:
            pass

    if data is None:
        start_brace = candidate_json.find("{")
        end_brace = candidate_json.rfind("}")
        if start_brace != -1 and end_brace > start_brace:
            try:
                data = json.loads(candidate_json[start_brace : end_brace + 1])
            except Exception:
                pass

    raw_items = []
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ["blocks", "layout", "elements", "chunks", "regions"]:
            if key in data and isinstance(data[key], list):
                raw_items = data[key]
                break
        if not raw_items:
            raw_items = [data]

    # 2. If JSON parsing found nothing, inspect HTML tags
    if not raw_items:
        raw_items = parse_html_bboxes(raw_response)

    # 3. If still empty, regex search for bbox and type patterns
    if not raw_items:
        regex_matches = re.findall(
            r'\{[^{}]*?"(?:bbox|box)"\s*:\s*\[([0-9.,\s]+)\][^{}]*?"(?:type|category|label)"\s*:\s*"([^"]+)"[^{}]*?\}',
            text,
            re.IGNORECASE,
        )
        if not regex_matches:
            regex_matches = re.findall(
                r'\{[^{}]*?"(?:type|category|label)"\s*:\s*"([^"]+)"[^{}]*?"(?:bbox|box)"\s*:\s*\[([0-9.,\s]+)\][^{}]*?\}',
                text,
                re.IGNORECASE,
            )
            regex_matches = [(b, t) for t, b in regex_matches]

        for bbox_str, type_str in regex_matches:
            coords = parse_bbox_values(bbox_str)
            if coords:
                raw_items.append({"bbox": coords, "type": type_str})

    # Standardize and filter detections
    detections = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        raw_type = item.get("type") or item.get("category") or item.get("label") or "Text"
        type_key = str(raw_type).strip().lower().replace("_", "-")
        norm_cat = CHANDRA_NORM_MAP.get(type_key, "Text")

        if norm_cat not in DOCLAYNET_CATEGORIES:
            continue

        cat_id = DOCLAYNET_CATEGORIES[norm_cat]
        raw_bbox = item.get("bbox") or item.get("box") or item.get("polygon")
        coords = parse_bbox_values(raw_bbox)
        if not coords:
            continue

        x1, y1, x2, y2 = coords

        # Scale detection
        if max(x1, x2) <= 1.05 and img_width > 1:
            x1 = x1 * img_width
            x2 = x2 * img_width
            y1 = y1 * img_height
            y2 = y2 * img_height
        elif max(x1, x2) <= 1000 and (img_width > 1000 or img_height > 1000):
            x1 = x1 * img_width / 1000.0
            x2 = x2 * img_width / 1000.0
            y1 = y1 * img_height / 1000.0
            y2 = y2 * img_height / 1000.0

        # Clip
        x1 = max(0.0, min(float(x1), float(img_width)))
        y1 = max(0.0, min(float(y1), float(img_height)))
        x2 = max(0.0, min(float(x2), float(img_width)))
        y2 = max(0.0, min(float(y2), float(img_height)))

        bx = min(x1, x2)
        by = min(y1, y2)
        bw = abs(x2 - x1)
        bh = abs(y2 - y1)

        if bw < 2 or bh < 2:
            continue

        score = float(item.get("score", item.get("confidence", default_score)))
        detections.append(
            {
                "category_id": cat_id,
                "category_name": norm_cat,
                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                "score": round(score, 4),
            }
        )

    return detections
