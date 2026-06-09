"""Generate the synthetic evaluation PDF set for the graphrag eval harness.

Produces four Korean fixture PDFs under tests/fixtures/eval_pdfs/, modeling
a fictitious public research institute "한국정책연구원 (KPRI)". The four
documents reference each other through entities (departments, titles,
people) and relations (reporting line, approval line, policy chain), which
gives GraphRAG something to traverse on multi-hop questions.

All names, amounts, deadlines, and rules are fictional and contain no PII.

Run inside the venv:
    .venv/bin/python scripts/build_eval_pdfs.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "eval_pdfs"

FONT_NAME = "HYSMyeongJo-Medium"


DOC_TRAVEL_POLICY = {
    "filename": "01_출장정책.pdf",
    "title": "한국정책연구원 출장 정책 규정",
    "body": [
        "1. 적용 범위",
        "   본 규정은 한국정책연구원(이하 'KPRI') 모든 정규직 연구원 및 행정직원의 국내 및 해외 출장에 적용된다.",
        "   계약직 연구원의 경우 5조의 별도 절차를 따른다.",
        "",
        "2. 출장 신청 절차",
        "   - 국내 출장: 출장 5일 전까지 소속 부서장의 승인을 받아야 한다.",
        "   - 해외 출장: 출장 14일 전까지 부서장 승인 후 행정지원실 사전 검토를 거친다.",
        "   - 1주일 이상의 장기 출장은 부원장의 최종 승인이 추가로 필요하다.",
        "",
        "3. 비용 처리 한도",
        "   - 국내 출장 일일 식비: 50,000원",
        "   - 해외 출장 일일 식비: 100 USD",
        "   - 국내 숙박비: 일 120,000원 한도 내 실비 정산",
        "   - 해외 숙박비: 일 200 USD 한도 내 실비 정산",
        "   - 항공권은 이코노미 클래스를 원칙으로 한다. 8시간 이상 비행 시 부원장 승인 하에 비즈니스 클래스 허용.",
        "",
        "4. 정산 담당",
        "   모든 출장비 정산은 행정지원실 회계팀(팀장: 박서연)에서 처리한다.",
        "   정산 서류는 영수증 원본과 함께 5조의 출장 보고서를 첨부하여 제출한다.",
        "",
        "5. 보고 의무",
        "   - 국내 출장: 종료 후 5일 이내에 출장 보고서를 제출해야 한다.",
        "   - 해외 출장: 종료 후 10일 이내에 출장 보고서와 결과 보고를 함께 제출한다.",
        "   - 양식과 작성 지침은 별도 문서 '출장 보고서 양식 및 작성 지침'을 참고한다.",
        "",
        "6. 위반 시 조치",
        "   사전 승인 없이 출장을 진행한 경우 출장비가 환수되며, 반복 위반 시 감사실에 회부된다.",
    ],
}


DOC_ORG_STRUCTURE = {
    "filename": "02_조직체계도.pdf",
    "title": "한국정책연구원 조직 체계 및 인원 현황",
    "body": [
        "1. 기관 개요",
        "   한국정책연구원(KPRI)은 원장 1인 - 부원장 1인 체제로 운영된다.",
        "   원장: 김도윤   부원장: 이지환",
        "",
        "2. 부서 구성 (5개 부서)",
        "   - 정책연구부: 부서장 최민호   소속 인원 18명",
        "   - 데이터분석부: 부서장 정수아   소속 인원 12명",
        "   - 국제협력부: 부서장 한태경   소속 인원 8명",
        "   - 행정지원실: 실장 윤하늘   소속 인원 9명 (회계팀 / 인사팀 / 총무팀)",
        "   - 감사실: 실장 강서윤   소속 인원 3명 (원장 직속)",
        "",
        "3. 보고 체계",
        "   - 정책연구부 / 데이터분석부 / 국제협력부 부서장 → 부원장 → 원장",
        "   - 행정지원실 실장 → 부원장 → 원장",
        "   - 감사실 실장 → 원장 (부원장을 경유하지 않는다)",
        "",
        "4. 행정지원실 회계팀",
        "   행정지원실 산하 회계팀의 팀장은 박서연이며, 인원은 3명이다.",
        "   회계팀은 모든 출장비 정산을 담당한다.",
        "",
        "5. 부재 시 대체 라인",
        "   - 원장 부재 시: 부원장이 모든 권한을 대행한다.",
        "   - 부원장 부재 시: 행정지원실 실장이 부원장 권한을 임시 대행한다.",
        "   - 부서장 부재 시: 해당 부서의 차상위 책임 연구원이 대행한다.",
    ],
}


DOC_APPROVAL_AUTHORITY = {
    "filename": "03_승인권한규정.pdf",
    "title": "한국정책연구원 결재 및 승인 권한 규정",
    "body": [
        "1. 목적",
        "   본 규정은 한국정책연구원(KPRI)의 출장 등 지출 행위에 대한 결재 권한 및 한도를 정한다.",
        "",
        "2. 출장비 승인 권한 (단일 출장 기준)",
        "   - 50만원 미만: 소속 부서장 전결",
        "   - 50만원 이상 200만원 미만: 부원장 결재",
        "   - 200만원 이상: 원장 최종 결재",
        "",
        "3. 해외 출장 추가 요건",
        "   해외 출장은 금액과 별개로 다음 절차를 거친다.",
        "   - 부서장 승인 후 행정지원실 실장의 사전 검토",
        "   - 부원장 최종 승인",
        "   - 1주일 이상 장기 해외 출장은 원장에게 사후 보고",
        "",
        "4. 감사실 출장",
        "   감사실 직원의 출장은 부원장을 경유하지 않고 원장이 직접 승인한다.",
        "   감사실 실장 강서윤의 출장은 원장의 단일 결재로 처리한다.",
        "",
        "5. 부원장 부재 시 대체 결재",
        "   부원장이 휴가, 출장, 병가 등으로 부재할 경우, 부원장 권한의 출장비 결재는",
        "   행정지원실 실장 윤하늘이 대행한다.",
        "   단, 200만원 이상 건은 대행 불가하며 원장에게 직접 상신한다.",
        "",
        "6. 긴급 출장",
        "   천재지변 또는 기관 운영상 긴급한 사유로 사전 승인이 불가능한 경우,",
        "   출장 후 3영업일 이내 사후 승인을 받는다. 단, 200만원 이상 출장은 긴급 사후 승인 대상에서 제외된다.",
    ],
}


DOC_REPORT_GUIDE = {
    "filename": "04_출장보고서지침.pdf",
    "title": "한국정책연구원 출장 보고서 양식 및 작성 지침",
    "body": [
        "1. 적용 범위",
        "   본 지침은 한국정책연구원 모든 출장자가 작성하는 출장 보고서에 적용된다.",
        "",
        "2. 보고서 구성 항목",
        "   - 출장자 인적사항: 성명, 소속 부서, 직급",
        "   - 출장 개요: 출장 기간, 출장지, 목적",
        "   - 주요 활동: 일자별 활동 내용",
        "   - 성과 및 시사점",
        "   - 후속 조치 계획",
        "   - 첨부: 영수증 및 증빙 자료",
        "",
        "3. 제출 기한",
        "   - 국내 출장: 출장 종료 후 5일 이내",
        "   - 해외 출장: 출장 종료 후 10일 이내 (결과 보고와 함께)",
        "",
        "4. 제출 경로",
        "   - 출장 보고서는 소속 부서장의 확인을 거쳐 행정지원실 회계팀에 제출한다.",
        "   - 회계팀(팀장: 박서연)은 보고서와 영수증을 대조하여 정산을 진행한다.",
        "",
        "5. 해외 출장 결과 보고",
        "   해외 출장 결과 보고는 출장 보고서와 별도로, 출장 목적별 성과 보고서를 작성하여",
        "   부원장에게 직접 보고한다. 1주일 이상 장기 출장은 원장 보고를 추가한다.",
        "",
        "6. 미제출 시 조치",
        "   기한 내 보고서 미제출 시 회계팀은 행정지원실 실장에게 통보하며,",
        "   정산은 보고서 제출 시까지 보류된다. 누적 3회 미제출 시 감사실 회부 대상이 된다.",
        "",
        "7. 보고서 보존",
        "   제출된 출장 보고서는 행정지원실에서 5년간 보존하며, 보존 기간 경과 후 폐기한다.",
    ],
}


DOCS = [DOC_TRAVEL_POLICY, DOC_ORG_STRUCTURE, DOC_APPROVAL_AUTHORITY, DOC_REPORT_GUIDE]


def render(doc: dict) -> Path:
    out = OUT_DIR / doc["filename"]
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4
    c.setFont(FONT_NAME, 16)
    c.drawCentredString(width / 2, height - 70, doc["title"])
    c.setFont(FONT_NAME, 11)
    y = height - 110
    for line in doc["body"]:
        if y < 80:
            c.showPage()
            c.setFont(FONT_NAME, 11)
            y = height - 70
        c.drawString(55, y, line)
        y -= 18
    c.showPage()
    c.save()
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    for doc in DOCS:
        out = render(doc)
        print(f"wrote {out.name}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
