# CADR — Confidence-Aware Domain Retrieval for Subtitle Post-processing

> **한 줄 요약**
> ASR/자막 후처리에서, 모델이 **불확실한 순간에만** 검색하고(confidence trigger),
> 검색 결과를 재생성이 아니라 **예측(logit) 단계에서 보정**하며(logit adjustment),
> 검색 범위는 **도메인 라우팅으로 좁혀**(Knowledge Router) 연산량과 지연시간을 함께 줄인다.

이 문서는 **허브(hub)** 다. 전체 흐름과 파트 간 관계만 정의하고,
각 파트의 세부 스펙은 아래 스포크(spoke) 파일에 있다.
문서 파일은 향후 코드 모듈과 1:1로 대응하도록 쪼개져 있다.

---

## 핵심 질문

> 모델의 내부 불확실성(confidence)을 Retrieval Trigger로 사용하고,
> 그 결과를 생성 이후가 아니라 **예측(logit) 단계에서 보정**할 경우,
> 기존 RAG 대비 **더 적은 Retrieval 호출로 동등하거나 더 높은 도메인 정확도**를
> 달성할 수 있는가?

세부 목표(G1–G4):

- **G1** Retrieval Trigger를 입력이 아닌 **모델 내부 상태(confidence)** 로 정의·검증.
- **G2** Retrieval 결과를 **logit bias** 로 주입해 재생성 없이 예측을 교정.
- **G3** Retrieval 호출 비율 대비 정확도의 **파레토 우위** 실증.
- **G4** 텍스트(MLM) 환경 검증 후 Whisper Decoder로 확장 가능한 일반 알고리즘 확립.

---

## 세 축의 절감 전략 (직교)

CADR의 효율성은 **서로 곱해지는 세 축**에서 나온다.

| 축 | 무엇을 정하는가 | 담당 파트 | 절감 대상 |
|---|---|---|---|
| **(1) 검색 여부** | 검색을 할지 말지 | `03_confidence_trigger` | 검색 **횟수** `ρ` |
| **(2) 검색 범위** | 검색한다면 어디를 볼지 | `01`, `02`, `04` | 검색 **1회당 비용** `C_ret` |
| **(3) 활성화 시점** | 언제부터 도메인 특화를 켤지 | `02_knowledge_router` | **cold-start 오라우팅** 방지 |

총비용 근사:

```
총비용 ≈ C_enc + ρ · C_ret(scope)

  ρ           ← (1) confidence trigger 로 축소
  C_ret(scope)← (2) 도메인 라우팅으로 N_all → N_domain 축소
  scope 활성화← (3) warm-up 이후에만 domain scope 적용
```

- (1)은 검색 **횟수**를 줄이고, (2)는 검색 **한 번의 비용**을 줄인다 → 곱셈으로 절감.
- (3)은 비용이 아니라 **정확도/안정성** 축이다. 문맥이 부족한 초반에 성급히 도메인을
  확정하면 오라우팅으로 오히려 정확도가 떨어지므로, **처음엔 general knowledge만**(warm-up,
  토큰 기준) 쓰고 라우터가 확신할 때 도메인 특화를 켠다. 단 초반부터 확신이 매우 높으면
  **Early Lock**으로 조기 진입(실시간성)하고, 한 번 lock된 도메인은 **히스테리시스**로
  작은 변화엔 유지해 잦은 전환을 막는다(→ `02`).

---

## 두 종류의 confidence (혼동 주의)

CADR에는 이름이 비슷하지만 **역할이 다른 두 confidence**가 있다. 파트가 다르다.

| | Router Confidence | Token Confidence |
|---|---|---|
| 질문 | "이 문서/발화가 **어느 도메인**인가?" | "이 **마스크 토큰**을 내가 아는가?" |
| 대상 | 문서 초반부 누적 근거 | 개별 예측 분포 `p` |
| 임계값 | `τ_dom` (도메인 lock-in) | `τ` (retrieval trigger) |
| 결과 | warm-up 종료 & domain scope 활성화 | 이 토큰에서 retrieval 수행 여부 |
| 파트 | `02_knowledge_router` | `03_confidence_trigger` |

---

## 전체 파이프라인 (Build-time vs Run-time)

```
[BUILD-TIME  (오프라인 전처리)]   ── 01_preprocessing_index
   Wiki/나무위키 문서
     → 카테고리 기반 규칙 태깅(Domain Tag)
     → 도메인별 분할 인덱스 (Sports / Game / AI / Cooking / Music)  + 도메인 용어집(FAISS)

────────────────────────────────────────────────────────────

[RUN-TIME  (디코더 입력부터, 토큰/스텝 단위 스트림)]

   토큰 스트림 t = 1,2,3, ...
        │
        ├─► Knowledge Router (02)  ── 매 스텝 도메인 근거 누적
        │       state = WARMUP | LOCKED(domains)
        │       WARMUP 조건: t < W  또는  라우터 미확신
        │       LOCK-IN:    top-k 도메인 점수 ≥ τ_dom  → scope = 그 도메인(들)
        │
        ▼
   scope = GENERAL (warm-up 중)  또는  LOCKED domain index(들) (lock 이후)
        │
        ▼
   Encoder(BERT/RoBERTa) → MLM logits z → p = softmax(z)
        │
        ▼
   Token Confidence c(p)        ── 03_confidence_trigger
        ├── c ≥ τ (High) ───────────────► output = argmax p          (검색 없음)
        └── c <  τ (Low)
                │
                ▼
          Retrieval R(context, scope) → bias b   ── 04_retrieval
                │   (scope 안에서만 검색; 부족하면 외부 dynamic retrieval)
                ▼
          Logit Adjustment z' = z + λ·b          ── 05_logit_adjustment
                ▼
          output = argmax softmax(z')
```

> **텍스트(Phase 1)에서의 "초반 30초" 정의**: 영상/음성이 아니라 **디코더 입력**부터
> 시작하므로, warm-up 창 `W`는 **문서 초반부**(앞쪽 N 토큰/문장)로 정의한다.
> Phase 2(Whisper)에서는 이 `W`가 자연스럽게 **실제 오디오 앞부분 ~30초**로 매핑된다
> (→ `07_roadmap`).

---

## 파일 맵

| 파일 | 내용 | 시점 | 향후 코드 모듈(예상) |
|---|---|---|---|
| `README.md` | 허브: 흐름·아키텍처·용어 | — | — |
| `01_preprocessing_index.md` | KB 구축·도메인 태깅·인덱스 분할 | Build | `indexer/`, `tagging/` |
| `02_knowledge_router.md` | 도메인 라우팅 + warm-up/lock-in | Run | `router/` |
| `03_confidence_trigger.md` | 토큰 confidence·trigger 수식 | Run | `trigger/` |
| `04_retrieval.md` | scope 검색 + dynamic retrieval | Run | `retrieval/` |
| `05_logit_adjustment.md` | 방식 A/B 보정 | Run | `adjust/` |
| `06_experiments.md` | 데이터셋·지표·baseline·ablation | — | `eval/` |
| `07_roadmap.md` | 일정·복잡도·Whisper 확장 | — | — |

`assemble.sh` 를 실행하면 위 파일을 순서대로 이어붙여 **단일 제출용 계획서**(`CADR_plan_full.md`)를 생성한다.

---

## 기존 연구 대비 위치 (요약)

| 항목 | Self-RAG | FLARE | Adaptive-RAG | **CADR (본 연구)** |
|---|---|---|---|---|
| Trigger 신호 | 학습된 reflection token | 미래 문장 저확신 토큰 | 입력 복잡도 분류기 | **현재 예측 confidence(margin/entropy)** |
| 보정 방식 | 재생성+self-critique | 재검색 후 재생성 | 경로 라우팅 후 생성 | **logit bias 주입(재생성 없음)** |
| 개입 지점 | 생성 | 생성 | 생성 | **예측(logit) 단계** |
| 검색 범위 | 전역 | 전역 | 전역 | **도메인 라우팅으로 축소** |
| 활성화 시점 | 항상 | 항상 | 항상 | **warm-up 후 도메인 확신 시** |

FLARE와 confidence-trigger 철학은 공유하되, CADR은 **(1) logit 보정, (2) 도메인 라우팅+용어집,
(3) warm-up 후 특화 활성화, (4) 무학습 이식 가능**의 결합에서 차별화된다.
