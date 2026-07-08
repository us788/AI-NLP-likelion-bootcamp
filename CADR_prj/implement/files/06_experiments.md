# 06 · 실험 설계 (데이터셋 · 지표 · Baseline · Ablation)

> **범위**: Phase 1(텍스트, BERT/RoBERTa MLM). 음성 입력 제외.

---

## 6.1 데이터셋

### 학습/인덱스 구축
- Wikipedia(일반), Reddit(구어/신조어), YouTube Transcript(자막 분포), 나무위키(용어 시드).
- 인덱스/용어집 구축 절차는 `01_preprocessing_index` 참조.

### 평가셋 (Ambiguity Benchmark)
애매한 표현을 포함한 문장을 별도 구축. 각 문장 = (문맥, 마스크 위치, 정답, 도메인 라벨, 혼동 후보).

예시:
- He got \_\_\_\_. → cooked (Game/Sports)
- The team used a high \_\_\_\_. → press (Football)
- Bro is \_\_\_\_. → cooking (meme)
- The model uses \_\_\_\_. → LoRA (AI)

도메인당 최소 300문장, 총 1,500문장 이상. 일반 도메인 대조군을 동일 규모로 포함.

### Warm-up 평가용 확장
warm-up/lock-in(신규)을 검증하려면 **문서 초반부의 도메인 모호성**이 있는 샘플이 필요하다.
- 도메인 단서가 **문서 뒤쪽**에 등장하는 긴 문맥 샘플을 별도 태깅.
- **다중 도메인 샘플**(예: 축구+게임)을 포함해 top-k(k=2) lock을 평가.

---

## 6.2 평가 지표

**정확도**
- Top-1 Accuracy, MLM Accuracy
- **Domain Term Accuracy** (도메인 용어 정답률, 핵심 지표)

**효율성**
- Retrieval 호출 비율 `ρ = (#retrieve)/(#tokens)`
- 평균 Latency(ms/token), 평균 추론 시간

**Router/Warm-up 지표(신규)**
- **Routing Accuracy**: lock된 `D*` 가 정답 도메인과 일치하는 비율.
- **Time-to-Lock**: warm-up 종료(=lock)까지 걸린 **토큰 수**(평균).
- **Early-Lock Rate**: `t < W` 에서 early lock된 비율(실시간성 지표).
- **Mis-routing Rate**: 잘못된 도메인으로 lock된 비율.
- **Domain Switch Count**: 문서당 평균 lock 해제/재-lock 횟수(히스테리시스로 낮을수록 안정).

**핵심 리포팅**: `Domain Term Accuracy` vs `ρ` 의 **파레토 곡선**.

---

## 6.3 Baselines

- **B0** Baseline BERT (retrieval 없음)
- **B1** BERT + Always Retrieval
- **B2** BERT + Confidence Retrieval (제안, CADR)
- **B2′** CADR **without warm-up**(라우팅을 t=0부터 강제) — warm-up 기여도 분리용(신규)

---

## 6.4 주요 실험

- **E1** B0/B1/B2 전체 비교(정확도·효율성).
- **E2** 도메인별 성능 분해(Sports/Game/AI/Cooking/Music).
- **E3** 파레토 분석: `τ` 스윕에 따른 (ρ, Domain Term Acc) 곡선.
- **E4** Warm-up 효과(신규): B2 vs B2′ 로 오라우팅·정확도 차이 측정.

---

## 6.5 Ablation Study

| Ablation | 변인 | 목적 |
|---|---|---|
| A1. Confidence metric | max / margin / entropy / weighted | 최적 token trigger 신호 |
| A2. Threshold `τ` | 0.1 ~ 0.9 스윕 | 정확도–비용 트레이드오프 |
| A3. Adjustment 방식 | Additive(A) vs Interpolation(B) | 보정 방식 효과 |
| A4. 보정 강도 `λ`/`β` | 0, 0.25, 0.5, 0.75, 1.0 | 과/과소보정 경계 |
| A5. Retrieval source | 용어집만 / 문서만 / 둘다 | 지식원 기여도 |
| A6. Domain routing | 유 vs 무(전역 검색) | 도메인 판별 가치 |
| **A7. Warm-up 창 `W`(토큰)** | 0 / 짧게 / 길게 | cold-start 회피 vs 특화 지연 |
| **A8. Base lock `τ_dom`** | 0.3 ~ 0.8 스윕 | 조기 lock vs 안정 lock |
| **A9. Top-k 도메인 수** | k=1 vs k=2 | 다중 도메인 lock의 이득 |
| **A10. Early lock `τ_early`** | early lock 유/무, 0.7~0.95 | 실시간성 vs 오라우팅 |
| **A11. 히스테리시스 `δ`,`m`** | δ∈{0,0.2,0.4}, m∈{1,8,16} | 도메인 진동(flapping) 억제 효과 |
| A12. Oracle 상한 | 정답 도메인·용어 주입 | 성능 상한(upper bound) |

(A7–A11이 이번에 추가된 Knowledge Router/warm-up/early-lock/히스테리시스 관련 신규 변인.)

---

## 6.6 통계적 검증
- 3-seed 평균 ± 표준편차.
- B1 대비 B2 정확도 차이는 paired bootstrap(≥1,000 resample)으로 유의성 검증.
