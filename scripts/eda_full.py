"""data/raw/documents/의 99개 문서 전체로 파싱/청킹 통계를 뽑아서 JSON으로 저장한다.

docs/eda.md, docs/parsing.md, docs/chunking_strategy.md 갱신용 원자료.
"""

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking.chunker import chunk_blocks
from src.ingestion.documents import extract_documents, parse_report_blocks

DOC_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "documents"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "eda_full_stats.json"


def percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    qs = statistics.quantiles(values, n=100) if len(values) >= 2 else [values[0]] * 99
    return {
        "min": min(values),
        "median": int(statistics.median(values)),
        "p90": int(qs[89]),
        "p95": int(qs[94]),
        "p99": int(qs[98]),
        "max": max(values),
        "n": len(values),
    }


def main():
    files = sorted(DOC_DIR.glob("*.zip"))

    per_doc = []
    all_para_lens = []
    all_table_lens = []
    all_chunk_lens = []
    chunks_per_doc = []
    top_section_counter = Counter()
    parse_errors = []

    for i, f in enumerate(files, 1):
        t0 = time.time()
        try:
            zip_bytes = f.read_bytes()
            docs = extract_documents(zip_bytes)
            blocks = []
            for _, content in sorted(docs.items()):
                blocks.extend(parse_report_blocks(content))
        except Exception as e:  # 전체 배치가 한 문서 때문에 죽지 않게
            parse_errors.append((f.name, repr(e)))
            print(f"[{i}/{len(files)}] {f.name}: 에러 - {e!r}", flush=True)
            continue

        type_counter = Counter(b["type"] for b in blocks)
        para_lens = [len(b["text"]) for b in blocks if b["type"] == "paragraph"]
        table_lens = [len(b["text"]) for b in blocks if b["type"] == "table"]
        all_para_lens.extend(para_lens)
        all_table_lens.extend(table_lens)

        for b in blocks:
            key = b["section_path"][0] if b["section_path"] else "(없음)"
            top_section_counter[key] += 1

        chunks = chunk_blocks(blocks)
        chunks_per_doc.append(len(chunks))
        all_chunk_lens.extend(len(c["text"]) for c in chunks)

        per_doc.append(
            {
                "file": f.name,
                "n_xml_parts": len(docs),
                "total_blocks": len(blocks),
                "n_paragraph": type_counter["paragraph"],
                "n_table": type_counter["table"],
                "total_chars": sum(len(b["text"]) for b in blocks),
                "n_chunks": len(chunks),
            }
        )
        print(f"[{i}/{len(files)}] {f.name}: {time.time() - t0:.2f}초, 블록 {len(blocks)}개, 청크 {len(chunks)}개", flush=True)

    result = {
        "n_documents": len(files),
        "n_parse_errors": len(parse_errors),
        "parse_errors": parse_errors,
        "paragraph_len_chars": percentiles(all_para_lens),
        "table_len_chars": percentiles(all_table_lens),
        "chunk_len_chars": percentiles(all_chunk_lens),
        "chunks_per_doc": percentiles(chunks_per_doc),
        "total_chunks_all_docs": len(all_chunk_lens),
        "top_section_distribution": top_section_counter.most_common(20),
        "per_doc": per_doc,
    }

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"{len(files)}개 문서 처리, 에러 {len(parse_errors)}건")
    print(f"저장: {OUTPUT}")


if __name__ == "__main__":
    main()
