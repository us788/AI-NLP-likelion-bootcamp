# 03 · Confidence Trigger (토큰 수준)

> **시점**: 런타임, 매 예측 스텝.
> **역할**: 이 토큰에서 **검색을 할지 말지**를 모델 내부 상태로 결정한다.
> 입력 텍스트의 복잡도가 아니라 **현재 예측 분포의 불확실성**이 신호다.

> **혼동 주의**: 이 파트의 임계값 `τ`(토큰 confidence)는
> `02_knowledge_router` 의 `τ_dom`(도메인 lock-in)과 **다른 신호**다.

---

## 3.1 표기

마스크(또는 디코딩) 위치의 어휘 분포를 `p = (p_1, ..., p_{|V|})`,
내림차순 정렬을 `p_(1) ≥ p_(2) ≥ ...` 라 하자.

---

## 3.2 Confidence 측정 후보

**(a) Maximum Probability**

$$
c_{\max}(p) = \max_{v} p_v = p_{(1)}
$$

**(b) Top1–Top2 Margin**

$$
c_{\text{margin}}(p) = p_{(1)} - p_{(2)}
$$

**(c) Entropy (정규화)**

$$
H(p) = -\sum_{v=1}^{|V|} p_v \log p_v,
\qquad
c_{\text{ent}}(p) = 1 - \frac{H(p)}{\log |V|} \in [0,1]
$$

엔트로피가 클수록(=불확실) `c_ent` 는 작아진다.

**(d) 결합 스코어 (선택)**

$$
c(p) = w_1\, c_{\max} + w_2\, c_{\text{margin}} + w_3\, c_{\text{ent}},
\qquad \textstyle\sum_i w_i = 1
$$

---

## 3.3 Trigger 규칙

$$
\text{Retrieve}(p) =
\begin{cases}
1 & \text{if } c(p) < \tau \\[4pt]
0 & \text{otherwise}
\end{cases}
$$

Entropy 단독 사용 시 부호가 반대이므로 `H(p) > τ_H` 로 트리거한다.

`τ` 는 (i) 고정 하이퍼파라미터, (ii) 검증셋에서 목표 retrieval 비율 `ρ` 에 맞춰 캘리브레이션,
(iii) 학습형 controller(후속)로 확장 가능.

---

## 3.4 의사코드 (Router 연동 포함)

```python
def predict_token(context, mask_pos, t, router, tau, lambda_):
    scope = router.update(context, t)     # 02: "General"(warm-up) or locked D*

    z = mlm_head(encoder(context))[mask_pos]   # logits [|V|]
    p = softmax(z)

    c = confidence(p)                     # c_max / c_margin / c_ent / weighted
    if c >= tau:
        return argmax(p)                  # High confidence: 검색 없음

    # Low confidence → scope 안에서만 검색 (04), warm-up 이면 General
    b = retrieve_bias(context, scope)     # bias vector [|V|]
    if scope == {"General"}:
        # warm-up: 도메인 특화 bias 미적용 (general/무보정) — 05 계약
        b = general_or_zero_bias(context)
    z_adj = z + lambda_ * b               # 05: logit adjustment
    return argmax(softmax(z_adj))
```

핵심: **검색 여부**는 `c(p)` 가, **검색 범위**는 `router.scope` 가 정한다(직교).
