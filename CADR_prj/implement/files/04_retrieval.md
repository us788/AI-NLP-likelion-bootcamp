# 04 · Retrieval (Scope 검색 + Dynamic Retrieval)

> **시점**: 런타임, `03` 에서 검색이 트리거된 경우에만 실행.
> **역할**: `02_knowledge_router` 가 정한 `scope` 안에서만 검색해 bias의 근거 후보를 만든다.
> 내부 지식으로 부족할 때만 외부 검색(dynamic retrieval)으로 확장한다.

---

## 4.1 Scope 기반 검색 (Domain Knowledge Activation)

검색은 전체 KB가 아니라 라우터가 활성화한 인덱스만 대상으로 한다.

```
scope = {"General"}           → General index만 조회 (warm-up 중)
scope = {Sports}              → Sports index만
scope = {Sports, AI}          → Sports index + AI index (다중 도메인)
```

이로써 ANN 검색 후보 수가 `N_all → N_domain` 으로 줄어 `C_ret` 이 감소한다(→ `07` 복잡도).

---

## 4.2 Dynamic Retrieval (외부 검색으로의 확장)

도메인 KB만으로 충분하면 외부 검색을 하지 않는다. 다음 **둘 중 하나** 일 때만 외부 검색:

1. 토큰 confidence 가 낮고(`03` 트리거) **且** 도메인 인덱스에서 충분한 근거를 못 찾음
   (검색 최고 유사도 < `τ_sim`, 또는 후보 부족).
2. 대상이 본질적으로 **최신/변화성** 정보:
   - 최신 신조어 · 최신 밈(meme)
   - 최신 인물 및 고유명사
   - 최신 기술 및 제품
   - 최신 도메인 지식

이렇게 대부분의 일반 추론은 도메인 KB만으로 처리하고, 지속 변화 정보에만 외부 검색을
수행하여 평균 검색 횟수와 latency를 줄인다.

```python
def retrieve_bias(context, scope):
    hits = search(indices_of(scope), query=embed(context))   # 도메인/General 인덱스
    if insufficient(hits):            # 최고 유사도 < τ_sim or 후보 부족
        hits += dynamic_external_search(context)  # 최신 신조어/밈/인물/기술 등
    return build_bias_vector(hits)    # → 05 로 전달되는 s_v/count 근거
```

---

## 4.3 검색 결과 → bias 근거

검색 결과 집합 `R` 로부터 각 어휘 `v` 의 **도메인 적합도 점수** `s_v`(문맥 임베딩과 후보 용어
임베딩 간 유사도) 또는 `count_R(v)` 를 만들어 `05_logit_adjustment` 로 넘긴다.

계약:

```
retrieve_bias(context, scope) -> b ∈ R^{|V|}   # 어휘별 bias 근거
```

> **warm-up 계약**: `scope == {"General"}` 인 동안에는 도메인 특화 bias를 만들지 않는다
> (general 근거만 사용하거나 `b=0`). 도메인 특화 bias는 lock 이후에만(→ `02`, `05`).
