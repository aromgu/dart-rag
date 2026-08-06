# 파싱 설계: DART 사업보고서 XML

관련 코드: [src/ingestion/documents.py](../src/ingestion/documents.py) · 테스트: [tests/test_report_parser.py](../tests/test_report_parser.py), [tests/test_ingestion.py](../tests/test_ingestion.py)

## 왜 새로 짰나

처음엔 `BeautifulSoup(xml_str).get_text()`로 태그를 전부 걷어내고 순수 텍스트만 뽑았다. 그런데 실제 문서(삼성전자 사업보고서 등)를 까보니:

- 문서 하나에 `TABLE` 태그가 최대 2,000개 이상 — 재무제표/임원현황 등 숫자 표가 태그 종류 기준으로 가장 큰 비중을 차지
- `get_text()`로 표까지 다 펼쳐버리면 표 안의 숫자들이 행/열 구분 없이 한 줄로 이어 붙어서, 어떤 숫자가 어떤 항목의 값인지 알 수 없는 텍스트가 됨
- 목차 구조(`SECTION-1/2/3`)가 있는데도 그냥 버려짐 — "이 문단이 재무제표 얘기인지 회사개요 얘기인지" 정보가 사라짐

이후 청킹 단계에서 표 중간이 잘리지 않게 하려면,애초에 파싱 단계에서 "이건 표다/문단이다"를 구분해서 갖고 있어야 한다. 그래서 DART의 실제 DTD 구조를 살려서 파싱하는 `parse_report_blocks()`를 새로 만들었다.

## DART XML 구조

DART 사업보고서류 원본(`document.xml` API 응답의 zip 안 파일)은 자체 DTD 기반 구조화 문서다. 삼성전자 사업보고서 태그 빈도(발췌):

```
TD 27,490    TE 16,149    P 15,627    TR 10,417    COL 5,673
TH 5,128     TBODY 2,027  TABLE 2,027 COLGROUP 2,027
SPAN 898     THEAD 613    TU 591      PGBRK 204
TABLE-GROUP 138  TITLE 135  SECTION-2 37  SECTION-1 14  SECTION-3 2
```

핵심 태그:

- `SECTION-1` / `SECTION-2` / `SECTION-3` — 목차 계층 (I장 > 절 > 항 형태로 중첩)
- `TITLE` — 각 SECTION의 제목 (`ATOCID` 속성으로 목차 순서 표시)
- `P` — 문단. `SPAN`으로 인라인 서식(볼드 등)이 걸리기도 하는데, `get_text()`가 알아서 하위 텍스트까지 다 모아주므로 별도 처리 불필요
- `TABLE` (`TABLE-GROUP`으로 감싸인 경우多) > `TBODY`/`THEAD` > `TR` > `TD`/`TE`/`TH`/`TU` — 표. `TU`는 `AUNIT`/`AUNITVALUE` 속성이 붙은 셀(예: 날짜, 금액 단위)로 보이나 텍스트 추출 관점에서는 다른 셀 태그와 동일하게 취급해도 무방

zip 안에는 XML이 여러 개 들어있을 수 있다 (회사/보고서에 따라 1~3개): 본문 1개 + 별도재무제표/연결재무제표 첨부(각각 "독립된 감사인의 감사보고서", "(첨부)재무제표" 등을 담은 별개 문서)가 붙는 경우가 있는데, 셋 다 같은 SECTION/TABLE 스키마를 쓴다.

## 블록 모델

`parse_report_blocks(xml_str) -> list[dict]`가 반환하는 각 블록:

```python
{"section_path": ["III. 재무에 관한 사항", "2. 연결재무제표"], "type": "paragraph" | "table", "text": "..."}
```

- `section_path`: 그 블록이 속한 `SECTION-1/2/3`의 `TITLE` 텍스트를 순서대로 쌓은 리스트. 최상위(커버 페이지 등, SECTION 밖)는 빈 리스트
- `type`: `paragraph`(P 태그) 또는 `table`(TABLE 태그)
- `text`: paragraph는 `get_text()` 결과, table은 아래 방식으로 직렬화한 텍스트

표는 행 단위로 셀을 `" | "`로 이어붙인다 (`_table_to_text`):

```
구분 | 2024 | 2023
매출액 | 1000 | 900
```

`ROWSPAN`/`COLSPAN`을 반영한 완전한 그리드 재구성은 하지 않는다 — RAG 컨텍스트로 쓰기에는 "한 행 안에서 셀들이 어떤 순서로 나열되는지"만 보존하면 충분하다고 판단, 과도한 엔지니어링을 피했다.

**중첩 표**: 표 셀 안에 표가 통째로 들어있는 경우가 실제로 있다(부문별 실적을 한 셀 안에 미니 표로 넣은 경우 등). 이걸 그냥 `get_text()`로 펴버리면 중첩 표의 모든 행이 구분 없이 한 줄로 뭉개져서 행 하나가 수만 자짜리가 되는 문제가 있었다(SK증권 사업보고서에서 발견). `_cell_text()`가 셀 안의 중첩 `TABLE`을 먼저 따로 직렬화해서 `[행1; 행2]` 형태로 대괄호로 붙이는 식으로 처리한다. 이때 바깥 `_table_to_text()`가 `table.find_all("TR")`(기본적으로 재귀적)로 중첩 표의 행까지 다시 주워오지 않도록, `tr.find_parent("TABLE") is table`로 "진짜 이 표에 직접 속한 행"만 걸러낸다.

`xml_to_text()`(기존 함수)는 이제 `parse_report_blocks()` 결과를 이어붙이는 방식으로 재구현되어 있어 파싱 로직이 하나로 통일되어 있다.

## 원본 XML 결함과 정제(`_sanitize_xml`)

DART가 내보내는 XML이 항상 완전한 well-formed XML은 아니었다. 99개 문서를 실제로 다 파싱해보며 발견한 문제 3가지와 대응:

### 1. 이스케이프 안 된 `&`

본문에 `Data & Solution`처럼 raw `&`가 그대로 들어있으면 XML 파서가 `no name` 에러를 내며 그 지점부터 태그 구조를 잃어버린다 (SK하이닉스 사업보고서에서 발견, `xmlParseEntityRef: no name` 에러).

**대응**: 이미 유효한 엔티티 참조(`&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&#123;`, `&#x1F;`)가 아닌 `&`만 골라서 `&amp;`로 미리 escape.

### 2. 장식용 꺾쇠괄호가 가짜 태그로 오인됨

소제목을 홑화살괄호(〈주요자회사〉) 대신 그냥 `<주요자회사>`로 써버린 경우가 다수 발견됨. XML Name 규칙상 한글도 태그 이름이 될 수 있어서, 파서는 이걸 진짜 여는 태그로 받아들였다가 대응하는 닫는 태그(`</주요자회사>`)가 없어서 "Opening and ending tag mismatch" 에러가 남 — 최악의 경우 뒤에 나오는 진짜 태그 구조를 다 잃어버리고 문서 나머지 전체가 하나의 거대한 문단(수십만 자)으로 뭉개졌다.

까다로웠던 점: `<SK스퀘어>`(영문자로 시작), `<Wholesale 부문>`(태그명+공백+속성처럼 보이는 형태)처럼 **문법적으로 진짜 태그와 구분이 안 되는 가짜 태그도 있어서**, "다음 글자가 영문자인가"나 "태그명 뒤에 공백이 오는가" 같은 문법 휴리스틱으로는 결국 다 못 걸렀다.

**대응**: 문법 휴리스틱을 포기하고 **화이트리스트**로 전환. 99개 문서 전체에서 실제 등장한 태그 이름 빈도를 세어보니(총 58개 후보), 진짜 DART 태그(수백~수백만 회 등장)와 장식용 꺾쇠(1~13회, 전부 회사/사업 용어)의 빈도 사이에 뚜렷한 단절이 있었다. 그렇게 뽑은 33개 태그만 화이트리스트에 넣고, 나머지는 태그명이 뭐든 전부 `&lt;`로 escape:

```python
_KNOWN_TAGS = {"TD", "TE", "TR", "COL", "TH", "P", "TBODY", "TABLE", ...}  # 33개, 전문은 documents.py 참고
```

### 3. `_sanitize_xml` 자체의 O(n²) 성능 버그

앞의 화이트리스트 로직을 처음 구현할 때 `rest = xml_str[m.end():]`로 "현재 위치부터 문서 끝까지"를 매번 새 문자열로 잘라냈다. 이러면 `<`를 만날 때마다 문서 나머지 전체를 복사하는 꼴이라, `<`가 수만~수십만 개인 대형 문서(TD/TE만 각각 수백만 회 등장)에서 사실상 O(n²)로 터진다 — 실제로 한 문서가 **몇 시간이 지나도 안 끝나는** 사고로 이어졌다(원래 20~30분이면 끝날 99개 문서 전체 EDA가 7시간 넘게 멈춰있었음).

**대응**: 부분문자열을 새로 만들지 않고 `re.match(string, pos)`의 `pos` 인자로 위치만 넘겨서 복사를 없앴다. 고친 뒤 6.5MB짜리 문제의 그 문서도 3초 만에 파싱됨. 회귀 방지용으로 "`<`가 4만 개인 문서를 2초 안에 처리하는지" 테스트도 추가했다(`test_sanitize_xml_stays_fast_on_documents_with_many_tags`).

### (참고) 정정 공시 3건의 문서가 아예 존재하지 않는 문제

파싱 버그는 아니지만 같이 발견한 문제: `document.xml` API가 파일이 없을 때 JSON이 아니라 `<result><status>014</status><message>파일이 존재하지 않습니다.</message></result>` 형태의 **XML 에러**를 돌려주는데, `DartClient.get_bytes()`가 Content-Type이 `json`인 경우만 에러로 걸러내던 탓에 이 XML 에러를 zip인 줄 알고 그대로 저장해버렸다. `client.py`에서 Content-Type 대신 **zip 매직바이트(`PK`)로 시작하는지**로 판별하도록 고쳐서 해결(에러 응답이면 JSON/XML 아무거나 파싱해서 status/message를 뽑아 `DartApiError`로 던짐). 실패했던 3건([첨부정정]/[기재정정]만 있고 본문 전체 재제출은 없는 정정 공시로 추정)은 같은 회사의 바로 이전 정기공시로 대체.

세 파싱 결함 정제는 모두 `parse_report_blocks()` 진입 시 자동 적용되며, 정제 자체는 대형 문서(5~6MB) 기준 0.1초 내외로 전체 파싱 시간(문서 복잡도에 따라 0.2~5초 수준)에 큰 영향을 주지 않는다.

## 공개 함수

| 함수 | 반환 | 용도 |
|---|---|---|
| `fetch_document_zip(rcept_no)` | zip bytes | 원본 다운로드만 |
| `extract_documents(zip_bytes)` | `{파일명: 텍스트}` | zip 안 XML들을 EUC-KR로 디코딩 |
| `parse_report_blocks(xml_str)` | `list[dict]` | 구조화된 블록 리스트 (청킹 입력으로 사용) |
| `xml_to_text(xml_str)` | `str` | 블록들을 이어붙인 단순 텍스트 (빠른 확인용) |
| `fetch_document_text(rcept_no)` | `str` | 원본→텍스트 한 번에 |
| `fetch_report_blocks(rcept_no)` | `list[dict]` | 원본→블록 한 번에 (여러 XML 파트는 이어붙임) |
