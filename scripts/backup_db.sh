#!/bin/bash
# postgres 전체 백업을 압축된 단일 파일로 뜬다.
# VM이 사라져도 데이터를 복구할 수 있도록, data/backups/에 저장한 뒤
# VSCode 탐색기 등으로 로컬 PC에 다운로드해서 보관하는 걸 권장한다.
#
# 복원: pg_restore -U dart_rag -h localhost -d dart_rag -c <파일명>
set -euo pipefail

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${BACKUP_DIR}/dart_rag_${TIMESTAMP}.dump"

echo "백업 시작: ${OUT_FILE}"
PGPASSWORD="${POSTGRES_PASSWORD:-dart_rag_dev_password}" pg_dump \
  -U "${POSTGRES_USER:-dart_rag}" \
  -h "${POSTGRES_HOST:-localhost}" \
  -d "${POSTGRES_DB:-dart_rag}" \
  -Fc -f "${OUT_FILE}"

echo "백업 완료: ${OUT_FILE} ($(du -h "${OUT_FILE}" | cut -f1))"
