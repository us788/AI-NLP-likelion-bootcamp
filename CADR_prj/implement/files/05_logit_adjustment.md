# 05 · Logit Adjustment (예측 단계 보정)

> **시점**: 런타임, 검색이 수행된 경우.
> **역할**: 검색 근거 `b` 를 **재생성 없이** logit에 반영해 예측을 교정한다.
> 이것이 CADR의 개입 지점 이동(생성 → 예측)의 핵심.

---

## 5.1 방식 A — Additive Logit Bias

검색 결과에서 bias를 정의한다.

$$
b_v = \log\!\big(\text{count}_R(v) + \alpha\big)
\quad\text{또는}\quad
b_v = \frac{\exp(s_v/T)}{\sum_{u} \exp(s_u/T)}
$$

`s_v` = 문맥·후보 임베딩 유사도, `T` = temperature, `α` = 스무딩 상수.
보정 logit과 최종 분포:

$$
z'_v = z_v + \lambda\, b_v,
\qquad
p'_v = \frac{\exp(z'_v)}{\sum_{u}\exp(z'_u)}
$$

`λ` 는 도메인 지식 주입 강도(`λ=0` 이면 baseline과 동일).

---

## 5.2 방식 B — 분포 보간 (kNN-LM style)

검색 분포 `p_R` 를 만들고 모델 분포와 보간한다.

$$
p_R(v) \propto \sum_{(k_i, v_i)\in R} \mathbb{1}[v_i = v]\,
\exp\!\Big(-\frac{\lVert q - k_i \rVert^2}{T}\Big)
$$

$$
p'(v) = (1-\beta)\, p_{\text{LM}}(v) + \beta\, p_R(v),
\qquad \beta \in [0,1]
$$

방식 B는 확률 공간 보간이라 해석이 직관적이고, 방식 A는 재정규화 없이 가산만으로 가능해
구현이 단순하다. **두 방식을 Ablation(A3)에서 비교한다.**

---

## 5.3 Warm-up 시의 보정 (Router 연동)

`02` 계약에 따라:

- `scope == {"General"}` (warm-up): **도메인 특화 bias 미적용**.
  `λ=0` 또는 general 근거만 반영 → 오라우팅으로 인한 잘못된 bias 주입 방지.
- `scope == D*` (locked): `D*` 용어집 근거로 `b`(또는 `p_R`) 구성 후 보정.

---

## 5.4 예시

| | press | price |
|---|---|---|
| 초기 `p` | 0.43 | 0.41 |
| Retrieval: `D*`=Sports, `b_press` ↑ | | |
| 보정 후 `p'` (λ 적용) | **0.62** | 0.30 |

→ 최종 선택 `press` (도메인 정합). warm-up 구간이었다면 이 보정은 적용되지 않고
초기 `p` 로 `price`/`press` 중 argmax 를 그대로 출력했을 것이다(오라우팅 회피의 반대급부).
