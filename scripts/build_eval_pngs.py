"""Generate the synthetic Korean evaluation set as PNG images.

Reportlab + CID Korean fonts produce PDFs whose glyphs Chandra OCR cannot
read (it returns "blank white page" for each one). PNGs with real
pixel-rendered Korean text using Noto Sans CJK work reliably.

Each input document is rendered as one A4-sized PNG; the ingest endpoint
accepts `.png` so we upload these directly.

Run inside the venv:
    .venv/bin/python scripts/build_eval_pngs.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from build_eval_pdfs import DOCS  # type: ignore

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eval_pdfs"

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

PAGE_W, PAGE_H = 1240, 1754
MARGIN = 80
TITLE_SIZE = 42
BODY_SIZE = 26
LINE_GAP = 14


def render(doc: dict) -> Path:
    out = OUT_DIR / doc["filename"].replace(".pdf", ".png")
    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(img)

    title_font = ImageFont.truetype(FONT_BOLD, TITLE_SIZE)
    body_font = ImageFont.truetype(FONT_REGULAR, BODY_SIZE)

    title = doc["title"]
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((PAGE_W - title_w) // 2, MARGIN), title, fill="black", font=title_font)

    y = MARGIN + TITLE_SIZE + 40
    for line in doc["body"]:
        if y > PAGE_H - MARGIN - BODY_SIZE:
            break
        draw.text((MARGIN, y), line, fill="black", font=body_font)
        y += BODY_SIZE + LINE_GAP

    img.save(out, format="PNG", optimize=True)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        out = render(doc)
        print(f"wrote {out.name}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
