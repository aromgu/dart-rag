"""검색/평가용 골드셋을 실제 DB에서 자동 구축한다.

기업별 손익계산서류 청크에서 "매출액"/"영업이익" 행을 정규식으로 찾아
값을 추출하고, 그 청크를 정답(ground truth)으로 하는 질문-정답 쌍을
data/eval/golden_qa.json에 저장한다. GPU 불필요 - SQL과 정규식만 쓴다.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.session import engine

OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_qa.json"

# 재무 수치는 보통 4자리 이상(콤마 포함)이라, 각주 번호(예: "30", "21") 같은
# 짧은 숫자를 값으로 잘못 뽑는 걸 막기 위해 최소 4자리를 요구한다.
_NUMBER = re.compile(r"\(?-?[\d,]{4,}\)?")
# "Ⅰ. 매    출    액"처럼 로마숫자/번호 접두어와 글자 사이 공백이 섞여 나오는
# DART 표기를 정규화하기 위한 접두어 패턴.
_PREFIX = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX0-9]*[.\)]?\s*")


def _normalize_label(cell: str) -> str:
    cell = _PREFIX.sub("", cell)
    return re.sub(r"\s+", "", cell)


def _extract_value(text_block: str, label: str, allow_negative: bool) -> str | None:
    """text_block에서 정규화한 첫 셀이 정확히 label로 시작하는 행을 찾아 값을 추출한다.

    "매출액"을 느슨한 부분일치(in)로 찾으면 "총매출액"/"내부매출액"/"매출원가" 같은
    다른 계정과목까지 잘못 걸려서(실제로 오리온에서 발생) 엉뚱한 값을 뽑는다 -
    정규화한 라벨이 target으로 *시작*하고 그 뒤엔 "(주석)" 각주 정도만 허용해서
    정확히 그 계정과목 행만 매칭되게 한다.
    """
    for line in text_block.split("\n"):
        cells = [c.strip() for c in line.split("|")]
        if not cells:
            continue
        norm = _normalize_label(cells[0])
        if not norm.startswith(label):
            continue
        remainder = norm[len(label):]
        if remainder and not remainder.startswith("("):
            continue  # "매출원가"처럼 label 뒤에 다른 글자가 바로 붙는 경우 제외
        for cell in cells[1:]:
            if not _NUMBER.fullmatch(cell):
                continue
            if not allow_negative and cell.startswith("("):
                continue  # 매출액은 구조적으로 음수일 수 없음(연결조정 등 다른 컬럼 오매칭 방지)
            return cell
    return None


def main() -> None:
    with engine.connect() as conn:
        companies = conn.execute(
            text("SELECT corp_code, corp_name FROM companies ORDER BY corp_name")
        ).fetchall()

        items = []
        for corp_code, corp_name in companies:
            rows = conn.execute(
                text(
                    """
                    SELECT c.id, c.text FROM chunks c
                    JOIN disclosures d ON c.rcept_no = d.rcept_no
                    WHERE d.corp_code = :corp_code
                      AND c.chunk_type = 'table'
                      AND c.text LIKE '%매출액%' AND c.text LIKE '%영업이익%'
                    """
                ),
                {"corp_code": corp_code},
            ).fetchall()

            for chunk_id, chunk_text in rows:
                revenue = _extract_value(chunk_text, "매출액", allow_negative=False)
                op_income = _extract_value(chunk_text, "영업이익", allow_negative=True)
                if revenue and op_income:
                    items.append(
                        {
                            "question": f"{corp_name}의 매출액은 얼마인가요?",
                            "corp_name": corp_name,
                            "expected_chunk_id": chunk_id,
                            "expected_value": revenue,
                            "metric": "매출액",
                        }
                    )
                    items.append(
                        {
                            "question": f"{corp_name}의 영업이익은 얼마인가요?",
                            "corp_name": corp_name,
                            "expected_chunk_id": chunk_id,
                            "expected_value": op_income,
                            "metric": "영업이익",
                        }
                    )
                    break  # 회사당 첫 번째로 매칭되는 손익계산서류 청크 하나만 사용

    OUTPUT.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"골드셋 {len(items)}개 문항 생성 ({len(items)//2}개 기업) -> {OUTPUT}")


if __name__ == "__main__":
    main()
