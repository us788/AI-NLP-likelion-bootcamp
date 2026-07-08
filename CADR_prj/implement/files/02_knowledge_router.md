# 02 · Knowledge Router (도메인 라우팅 + Warm-up/Lock-in)

> **시점**: 런타임. 토큰/스텝 스트림과 병행하여 동작.
> **역할**: (a) 입력이 **어느 도메인**인지 판별하고, (b) **언제부터** 도메인 특화 기능을
> 켤지 결정한다. 기존 계획서의 "Domain Analyzer / Domain Classifier `D(context)`" 를
> 이 **Knowledge Router** 개념으로 대체·확장한다.

명칭 변경 이유: 기존 `Domain Analyzer` 는 "입력을 분석해 라벨을 붙인다"는 정적 뉘앙스였다.
본 파트는 라벨링에 더해 **어떤 지식 소스(scope)를 활성화할지 라우팅**하고, **상태를 전이**
(warm-up → locked)시키는 능동적 모듈이므로 **Knowledge Router** 가 더 정확하다.

---

## 2.1 왜 warm-up이 필요한가 (Cold-start 문제)

디코딩 초반에는 문맥이 거의 없어서 도메인 근거가 부족하다.
이때 성급히 도메인을 확정하면 **오라우팅(mis-routing)** 이 발생하고, 잘못된 도메인 용어집이
logit bias로 주입되어 오히려 정확도가 **떨어진다**(hallucination 재주입).

→ 해결: **초반에는 General knowledge만으로 돌리다가, 도메인이 확실해지면 특화 기능을 켠다.**

**Warm-up 창 `W` 는 문장 수가 아니라 토큰(Token) 수로 정의한다.**
자막 시스템은 문장 경계가 불명확하고 플랫폼·STT 방식에 따라 문장 분할이 달라지는 반면,
토큰은 모델 내부에서 일관된 단위로 쓰이므로 더 안정적인 기준이 된다.

- 텍스트(Phase 1): "초반"은 **디코더 입력의 앞쪽 `W` 토큰**.
  영상/음성이 아니라 **디코더 입력**부터 시작하므로 문서 초반부(토큰 기준)로 정의한다.
- 음성(Phase 2, Whisper): 이 `W` 토큰 창이 자연스럽게 **실제 오디오 앞 ~30초** 에 매핑된다(→ `07`).

---

## 2.2 라우터 상태 (State Machine)

```
        ┌─────────────┐
        │   WARMUP     │   scope = General, 도메인 근거 S[d] 누적
        │ (general만)  │
        └──────┬───────┘
               │  Base Lock ( t≥W ∧ maxS ≥ τ_dom )
               │  Early Lock( t<W ∧ maxS ≥ τ_early ),  τ_early > τ_dom
               ▼
        ┌─────────────┐
        │  LOCKED(D*)  │   scope = D* (1개 또는 top-k 다중 도메인)
        │ 도메인 특화 켜짐│  domain index + logit bias 활성
        └──────┬───────┘
               │  Hysteresis Unlock
               │  ( 반대도메인 우세 ≥ δ 가 연속 m 토큰 지속 )
               ▼
        (WARMUP 복귀 → 재-lock)
```

- **WARMUP**: `scope = General`. retrieval이 트리거돼도 General 인덱스만 조회.
  도메인 특화 logit bias는 **적용하지 않는다**.
- **LOCKED(D\*)**: `scope = D*`. `D*` 는 확정된 도메인 집합(단일 또는 다중).
  이후 검색·bias가 `D*` 인덱스/용어집으로 한정된다.

---

## 2.3 도메인 근거 누적 & Lock-in 규칙

Knowledge Router는 **Multi-label** 로 도메인을 판별한다(단일 분류 아님).
스텝 `t` 까지의 문맥으로 도메인 `d` 에 대한 순간 점수 `g_t(d) ∈ [0,1]` 를 얻고,
이를 누적해 안정적인 도메인 신뢰도 `S_t(d)` 를 만든다.

**누적(EMA):**

$$
S_t(d) = \gamma\, S_{t-1}(d) + (1-\gamma)\, g_t(d),
\qquad \gamma \in [0,1)
$$

**Top-k 선택 & Lock-in 조건.** 세 가지 규칙으로 도메인 특화를 켜고 끈다.

**(1) 기본 Lock (Base Lock)** — 성급한 고정 방지.
warm-up 종료(`t ≥ W`) 이후, 도메인 신뢰도가 임계값 `τ_dom` 이상일 때 lock.

$$
\text{Base Lock} \iff \Big( t \ge W \Big) \ \wedge\ \Big( \max_d S_t(d) \ge \tau_{\text{dom}} \Big)
$$

**(2) 조기 Lock (Early Lock)** — 실시간성 확보.
모델이 warm-up이 끝나기 전이라도 **매우 높은 확신**을 보이면 기다리지 않고 lock.
더 엄격한 임계값 `τ_early` (단, `τ_early > τ_dom`)를 사용한다.

$$
\text{Early Lock} \iff \Big( t < W \Big) \ \wedge\ \Big( \max_d S_t(d) \ge \tau_{\text{early}} \Big)
$$

즉 최종 lock 조건은 `Base Lock ∨ Early Lock`. `τ_early > τ_dom` 로 둠으로써,
문맥이 짧은 초반에는 어지간한 확신으로는 lock하지 않고 **확실할 때만** 조기 진입한다.

lock 시점의 활성 도메인 집합(top-k, 다중 허용):

$$
D^\* = \{\, d \ :\ S_t(d) \ge \tau_{\text{dom}} \,\}\ \ \text{(최대 } k \text{개)}
$$

- **단일 도메인**: 예) `S(Sports)=0.88` 하나만 초과 → `D* = {Sports}`.
- **다중 도메인(허용)**: 예) `S(Sports)=0.84, S(Game)=0.79` 둘 다 초과 → `D* = {Sports, Game}`
  (예: "축구 게임 방송" 처럼 축구+게임 동시 활성).

**(3) 히스테리시스 해제 (Hysteresis Unlock)** — 불필요한 전환 방지.
한 번 lock된 도메인은 작은 confidence 변화만으로 즉시 바꾸지 않는다.
현재 `D*` 밖 도메인의 신뢰도가 현재 도메인을 **`δ` 이상 우세**한 상태가
**연속 `m` 토큰** 지속될 때만 해제한다(둘 다 만족해야 함).

$$
\text{Unlock} \iff
\Big( \max_{d \notin D^\*} S_t(d) - \max_{d \in D^\*} S_t(d) \ge \delta \Big)
\ \text{가 연속 } m \text{ 토큰 지속}
$$

- Lock 임계 `τ_dom` 과 Unlock 마진 `δ` 를 분리해 둔 것이 히스테리시스의 핵심
  (진입은 `τ_dom`, 이탈은 `δ`+지속시간) → 경계 근처에서의 잦은 도메인 진동(flapping)을 막는다.
- 해제되면 짧은 재적응을 위해 `WARMUP` 로 복귀 후 재-lock(→ 2.5 drift/재라우팅).

---

## 2.4 예시

```
입력 : "손흥민이 ChatGPT를 이용하여 전술을 분석했다."

Knowledge Router (누적 후)
  Sports : 0.87
  AI     : 0.82
  News   : 0.24
      ↓  τ_dom = 0.5, top-k=2
  D* = { Sports, AI }        → 두 도메인 index 동시 활성
```

warm-up 관점의 시간 축 예시(토큰 기준):

```
[기본]  t=1..W-1 : scope=General   ("어제 있었던 일인데" → 도메인 불명)
        t=W      : S(Sports)=0.86 ≥ τ_dom → Base Lock, scope={Sports}
        t>W      : Sports index + 용어집으로 검색/보정 활성

[조기]  t=6 (<W) : S(AI)=0.94 ≥ τ_early → Early Lock, scope={AI}
                    (강한 초반 단서 "LoRA fine-tuning" 등 → 실시간 진입)

[해제]  lock 후 반대 도메인 우세(≥δ)가 연속 m 토큰 → Hysteresis Unlock → WARMUP 복귀
```

---

## 2.5 Drift / 재라우팅 (히스테리시스 기반)

한 문서 안에서 주제가 바뀔 수 있다(예: 스포츠 뉴스 → 요리 코너).
해제는 2.3(3)의 **히스테리시스 규칙** 을 따른다 — 작은 변화로 즉시 바꾸지 않는다.

- 감지: `max_{d∉D*} S_t(d) − max_{d∈D*} S_t(d) ≥ δ` 가 **연속 `m` 토큰** 지속.
- 대응: `WARMUP` 로 잠시 복귀(짧은 재적응) 후 새 `D*` 로 재-lock.
- Phase 1 기본 실험은 문서당 단일 lock 을 가정하고, 재라우팅은 별도 실험(다중 주제 문서)에서 검증.

---

## 2.6 의사코드

```python
class KnowledgeRouter:
    def __init__(self, W, tau_dom, tau_early, k, gamma, delta, m):
        self.W = W                 # warm-up window (토큰 수)
        self.tau_dom = tau_dom     # base lock 임계값
        self.tau_early = tau_early # early lock 임계값 (> tau_dom)
        self.k = k                 # 최대 활성 도메인 수 (다중 허용)
        self.gamma = gamma         # EMA 계수
        self.delta = delta         # 히스테리시스 unlock 마진
        self.m = m                 # unlock 지속 토큰 수
        self.state = "WARMUP"
        self.scope = {"General"}
        self.S = defaultdict(float)  # 누적 도메인 신뢰도
        self._unlock_run = 0         # 반대증거 연속 카운터

    def _select(self):
        # 신뢰도 내림차순 중 tau_dom 이상인 상위 k개 도메인
        ranked = sorted(self.S, key=self.S.get, reverse=True)
        return set([d for d in ranked if self.S[d] >= self.tau_dom][:self.k])

    def update(self, context, t):
        g = domain_scores(context)          # {domain: [0,1]}  multi-label
        for d, gd in g.items():
            self.S[d] = self.gamma * self.S[d] + (1 - self.gamma) * gd
        top_s = max(self.S.values(), default=0.0)

        if self.state == "WARMUP":
            base_lock  = (t >= self.W) and (top_s >= self.tau_dom)
            early_lock = (t <  self.W) and (top_s >= self.tau_early)
            if base_lock or early_lock:
                self.state, self.scope = "LOCKED", self._select()

        elif self.state == "LOCKED":
            in_best  = max((self.S[d] for d in self.scope), default=0.0)
            out_best = max((self.S[d] for d in self.S if d not in self.scope),
                           default=0.0)
            # 히스테리시스: 반대증거가 연속 m 토큰 지속될 때만 해제
            self._unlock_run = self._unlock_run + 1 \
                if (out_best - in_best) >= self.delta else 0
            if self._unlock_run >= self.m:
                self.state, self.scope, self._unlock_run = "WARMUP", {"General"}, 0

        return self.scope     # {"General"} (warm-up) or locked domain set
```

- `domain_scores()` 는 규칙 기반(키워드/엔티티 매칭) 또는 경량 multi-label 분류기.
  Phase 1은 무학습 규칙 baseline으로 시작하고, 분류기 버전을 ablation으로 비교(→ `06`).

---

## 2.7 하이퍼파라미터 & 다른 파트와의 계약

| 파라미터 | 의미 | 기본값(초기) |
|---|---|---|
| `W` | warm-up 창(**토큰 수**) | 코퍼스 통계로 캘리브레이션 |
| `τ_dom` | base lock 임계값 | 0.5 (스윕) |
| `τ_early` | early lock 임계값 (`> τ_dom`) | 0.8 (스윕) |
| `k` | 최대 활성 도메인 수 | 2 |
| `γ` | 근거 누적 EMA 계수 | 0.8 |
| `δ` | 히스테리시스 unlock 마진 | 0.2 (스윕) |
| `m` | unlock 지속 토큰 수 | 8 (스윕) |

계약(출력):

```
router.scope ∈ { {"General"}, D* ⊆ {Sports,Game,AI,Cooking,Music} }
```

- `04_retrieval` 은 이 `scope` 안에서만 검색한다.
- `05_logit_adjustment` 은 `scope == {"General"}` (warm-up) 인 동안 **도메인 bias를 적용하지 않는다**
  (general bias/무보정). lock 이후에만 도메인 특화 bias 주입.

> **혼동 주의**: 여기서 다루는 `τ_dom`(도메인 lock-in)은
> `03_confidence_trigger` 의 토큰 confidence 임계값 `τ`(검색 여부)와 **다른 신호**다.
> 전자는 "어느 도메인인가", 후자는 "이 토큰을 아는가".
