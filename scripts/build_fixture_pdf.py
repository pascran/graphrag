"""Generate the deterministic Korean fixture PDF for the E2E ingest test.

Run from the repo root:

    .venv-test/bin/python scripts/build_fixture_pdf.py

Produces tests/fixtures/sample_pdfs/korean_form.pdf — a 1-page synthetic
travel-policy document. Content is fictitious and contains no PII.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_pdfs" / "korean_form.pdf"

TITLE = "사내 출장 정책 안내서"
LINES = [
    "1. 적용 범위",
    "   본 정책은 모든 정규직 임직원의 국내 및 해외 출장에 적용됩니다.",
    "",
    "2. 출장 신청 절차",
    "   - 출장 7일 전까지 부서장 승인을 받아야 합니다.",
    "   - 해외 출장은 14일 전 인사팀에 사전 통보합니다.",
    "",
    "3. 비용 처리 한도",
    "   - 국내 출장 일일 식비: 50,000원",
    "   - 해외 출장 일일 식비: 100 USD",
    "   - 숙박비는 영수증 기반 실비 정산을 원칙으로 합니다.",
    "",
    "4. 보고 의무",
    "   출장 종료 5일 이내에 출장 보고서를 제출해야 합니다.",
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))

    c = canvas.Canvas(str(OUT), pagesize=A4)
    width, height = A4
    c.setFont("HYSMyeongJo-Medium", 18)
    c.drawCentredString(width / 2, height - 80, TITLE)

    c.setFont("HYSMyeongJo-Medium", 12)
    y = height - 130
    for line in LINES:
        c.drawString(60, y, line)
        y -= 20

    c.showPage()
    c.save()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
