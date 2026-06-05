"""Chandra OCR 2 wrapper.

Strategy:
- PDF -> rasterize to PNG per page via pdf2image (poppler).
- Image -> just open and base64.
- Send each page to Chandra via OpenAI-compatible /v1/chat/completions
  with image_url data URI; ask for raw markdown of the page.

The PLAN says Chandra also produces structured JSON for form documents,
but for the MVP we ask for markdown which preserves tables. Form-level
JSON extraction is a follow-up enhancement.
"""
from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image
from pdf2image import convert_from_path
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("app.ingest.ocr")

OCR_PROMPT = (
    "Transcribe this document page exactly as Markdown. "
    "Preserve tables as Markdown tables, lists as lists, headings with #. "
    "Do not summarise. Do not add commentary. Output Markdown only."
)


@dataclass(frozen=True)
class OcrPage:
    page_number: int  # 1-indexed
    markdown: str


def _pil_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _ocr_image_data_uri(client: httpx.AsyncClient, model: str, data_uri: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    r = await client.post("/chat/completions", json=payload, timeout=600.0)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


async def ocr_pdf(path: str | Path, dpi: int = 200) -> list[OcrPage]:
    settings = get_settings()
    pages = convert_from_path(str(path), dpi=dpi)
    log.info("ocr_pdf_start", path=str(path), pages=len(pages), dpi=dpi)

    out: list[OcrPage] = []
    async with httpx.AsyncClient(base_url=settings.vllm_ocr_url) as client:
        for i, img in enumerate(pages, start=1):
            data_uri = _pil_to_data_uri(img)
            md = await _ocr_image_data_uri(client, settings.vllm_ocr_model, data_uri)
            out.append(OcrPage(page_number=i, markdown=md))
            log.info("ocr_page_done", page=i, chars=len(md))
    return out


async def ocr_image(path: str | Path) -> list[OcrPage]:
    settings = get_settings()
    data_uri = _file_to_data_uri(Path(path))
    async with httpx.AsyncClient(base_url=settings.vllm_ocr_url) as client:
        md = await _ocr_image_data_uri(client, settings.vllm_ocr_model, data_uri)
    return [OcrPage(page_number=1, markdown=md)]


async def ocr_file(path: str | Path) -> list[OcrPage]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return await ocr_pdf(p)
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        return await ocr_image(p)
    raise ValueError(f"unsupported file type for OCR: {suffix}")
