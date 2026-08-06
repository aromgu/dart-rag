from src.ingestion.client import DartClient
from src.ingestion.exceptions import DartApiError


def search_disclosures(
    client: DartClient | None = None,
    *,
    corp_code: str | None = None,
    bgn_de: str | None = None,
    end_de: str | None = None,
    pblntf_ty: str | None = None,
    pblntf_detail_ty: str | None = None,
    corp_cls: str | None = None,
    sort: str = "date",
    sort_mth: str = "desc",
    page_count: int = 100,
) -> list[dict]:
    """DART list.json 엔드포인트로 공시 목록을 검색하며, 모든 페이지를 끝까지 순회한다.

    bgn_de/end_de는 YYYYMMDD 형식이다. pblntf_ty는 공시유형 코드
    (예: "A" 정기공시, "B" 주요사항보고 ...). 전체 코드는 DART Open API 문서 참고.
    """
    client = client or DartClient()
    page_no = 1
    results: list[dict] = []

    while True:
        try:
            data = client.get_json(
                "list.json",
                corp_code=corp_code,
                bgn_de=bgn_de,
                end_de=end_de,
                pblntf_ty=pblntf_ty,
                pblntf_detail_ty=pblntf_detail_ty,
                corp_cls=corp_cls,
                sort=sort,
                sort_mth=sort_mth,
                page_no=page_no,
                page_count=page_count,
            )
        except DartApiError as e:
            if e.status == "013":  # 조회된 데이타가 없습니다
                break
            raise

        results.extend(data.get("list", []))

        total_page = int(data.get("total_page", 1))
        if page_no >= total_page:
            break
        page_no += 1

    return results
