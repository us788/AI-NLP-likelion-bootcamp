#!/usr/bin/env bash
# 스포크 파일들을 순서대로 이어붙여 단일 제출용 계획서를 생성한다.
# 사용법:  bash assemble.sh   →   CADR_plan_full.md 생성
set -euo pipefail
cd "$(dirname "$0")"

OUT="CADR_plan_full.md"
FILES=(
  "README.md"
  "01_preprocessing_index.md"
  "02_knowledge_router.md"
  "03_confidence_trigger.md"
  "04_retrieval.md"
  "05_logit_adjustment.md"
  "06_experiments.md"
  "07_roadmap.md"
)

: > "$OUT"
for f in "${FILES[@]}"; do
  cat "$f" >> "$OUT"
  printf '\n\n---\n\n' >> "$OUT"
done

echo "생성 완료: $OUT ($(wc -l < "$OUT") lines)"
