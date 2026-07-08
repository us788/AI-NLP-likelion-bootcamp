# 아이디어 발전 과정 (Argument Log)

## 1. 문제 정의

초기 아이디어는 **유튜브 자막 시스템의 정확도를 향상시키는 방법**이었다.

기존 자막 시스템은 일반적인 문장에서는 높은 정확도를 보이지만,

* 신조어
* 밈(Meme)
* 스포츠 용어
* 게임 용어
* 최신 기술 용어

와 같은 도메인 특화 표현에서 오류가 발생하는 경우가 많다.

특히 유튜브는 특정 주제에 대한 영상이 많기 때문에, 일반 언어 모델보다 **도메인에 특화된 지식**이 필요하다고 판단하였다.

---

# 2. 초기 아이디어

초기에는 단순히 RAG(Retrieval-Augmented Generation)를 이용하여 부족한 정보를 검색하는 방식을 고려하였다.

그러나 이 방법은 다음과 같은 문제가 있었다.

* 대부분의 문장은 검색이 필요하지 않다.
* 모든 문장에서 Retrieval을 수행하면 비용이 증가한다.
* 실시간 자막 시스템에서는 Retrieval로 인한 Latency가 커질 수 있다.

따라서

> "언제 Retrieval을 수행할 것인가?"

가 핵심 문제가 되었다.

---

# 3. Confidence-aware Retrieval

이후 Retrieval의 기준을 입력 문장이 아니라

**모델의 내부 상태(Model Confidence)**

로 결정하는 방향으로 발전하였다.

기본 아이디어는

```
모델 추론

↓

Confidence 계산

↓

확신이 높다

→ 그대로 출력

확신이 낮다

→ Retrieval 수행
```

이다.

이 구조라면 Retrieval은

정말 필요한 경우에만 수행된다.

---

# 4. Retrieval 위치에 대한 고민

초기에는

Softmax 이후의 확률분포를 이용하여

Retrieval 여부를 결정하는 구조를 고려하였다.

이후

Logits를 직접 수정하는 방식도 검토하였다.

```
Decoder

↓

Logits

↓

Confidence

↓

Retrieval

↓

Logit Adjustment

↓

Softmax

↓

Output
```

다만

Logits 자체는 절대적인 의미를 가지지 않으므로

Confidence 계산은 Softmax 이후의 확률분포를 사용하는 것이 더 적절하다는 결론에 도달하였다.

---

# 5. Beam Search 활용 여부

Beam Search를 이용하여

후보 문장을 여러 개 생성한 뒤

RAG를 이용하여 재정렬(Re-ranking)하는 방법도 고려하였다.

그러나

```
후보 생성

↓

RAG

↓

Re-ranking
```

구조는

기존 RAG 기반 시스템과의 차별성이 부족하다고 판단하였다.

따라서

Beam Search 기반 후처리보다는

모델이 추론하는 도중

불확실성을 감지하는 구조가 더 적합하다고 판단하였다.

---

# 6. 음성 모델을 직접 수정할 것인가?

Whisper와 같은 ASR 모델의 Decoder를 직접 수정하는 방법도 검토하였다.

하지만

* 음성 데이터 확보가 어렵고
* 학습 비용이 매우 크며
* 실험 반복 속도가 느리다.

따라서

초기 연구에서는

음성 입력을 제외하고

텍스트 환경에서 알고리즘을 검증하는 방향으로 변경하였다.

---

# 7. Text 기반 검증

음성을 제거하면

다음과 같은 구조가 된다.

```
문장

↓

BERT

↓

Softmax

↓

Confidence

↓

Retrieval

↓

Logit Adjustment

↓

Output
```

이 구조를 통해

핵심 아이디어인

Confidence-aware Retrieval

만 독립적으로 검증할 수 있다.

이후

동일한 알고리즘을

Whisper Decoder에 적용하는 것을 목표로 한다.

---

# 8. Domain Information 활용

이후

단순히 Retrieval만 수행하는 것이 아니라

입력 데이터의

**도메인(Domain)**

을 먼저 분석하는 방향으로 발전하였다.

예를 들어

* Sports
* Game
* AI
* Music
* Cooking

등으로 분류한다.

---

# 9. Domain Knowledge Base

범용적인 Knowledge Base를 구축하고

모든 문서를

도메인별로 태깅한다.

초기 구현에서는

별도의 분류 모델을 만들지 않고

위키의 Category 정보를 이용하여

Rule-based Tagging을 수행한다.

예시

```
손흥민

↓

Sports
```

```
ChatGPT

↓

AI
```

---

# 10. Domain Activation

도메인이 결정되면

전체 Knowledge Base를 사용하는 것이 아니라

해당 도메인의 정보만 활성화한다.

예시

```
Knowledge Base

Sports

AI

Game

Music

↓

Domain = Sports

↓

Sports Knowledge만 사용
```

이를 통해

검색 공간을 줄일 수 있다.

---

# 11. Dynamic Retrieval

도메인 Knowledge만으로

문제를 해결할 수 있다면

Retrieval은 수행하지 않는다.

반대로

* Confidence가 낮거나
* Domain Knowledge에 존재하지 않는 정보라면

Dynamic Retrieval을 수행한다.

검색 대상은

* 최신 밈
* 최신 신조어
* 최신 인물
* 최신 기술
* 최신 도메인 지식

등이다.

---

# 12. 최종 구조

최종적으로 현재 구상하는 시스템은 다음과 같다.

```
Input

↓

Domain Analyzer

↓

Domain Knowledge Activation

↓

Model Inference

↓

Confidence Estimation

↓

Confidence High

↓

Output

----------------

Confidence Low

↓

Dynamic Retrieval

↓

Logit Adjustment

↓

Output
```

---

# 13. 핵심 차별점

기존 RAG는

입력을 기준으로 Retrieval을 수행한다.

본 아이디어는

모델의 내부 상태(Confidence)를 기준으로

Retrieval 여부를 결정한다.

또한

전체 Knowledge Base를 검색하지 않고

도메인을 먼저 분석하여

검색 공간을 제한한 뒤

필요한 경우에만 Dynamic Retrieval을 수행한다.

즉,

Retrieval을 많이 수행하는 것이 아니라

**언제, 어디에서 Retrieval을 수행할 것인가**를 최적화하는 것이 본 연구의 핵심 아이디어이다.

---

# 향후 검토할 사항

* Confidence Estimation 방법

  * Top-1 / Top-2 Margin
  * Entropy
  * Temperature Scaling
  * 학습 기반 Retrieval Controller

* Domain Analyzer

  * Rule-based
  * Multi-label Classification

* Logit Adjustment 방식

  * 단순 Bias
  * 학습 가능한 Bias
  * Domain Prior

* Retrieval Trigger

  * Threshold 기반
  * Learning-based

* Whisper Decoder 적용 가능성

* Latency 및 Retrieval 비용 분석

* 기존 Adaptive RAG, Self-RAG, Retrieval-on-Demand와의 차별성 분석
