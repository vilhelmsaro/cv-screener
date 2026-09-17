"""Small text helpers shared by the CV models and the field rules."""
from __future__ import annotations


def split_top_level(text: str, separators: str = ",;") -> list[str]:
    """Split on separators outside parentheses: "AWS (EC2, S3), Go" -> ["AWS (EC2, S3)", "Go"]."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch in separators and depth <= 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]
