# AI-NLP-likelion-bootcamp

# CADR (Confidence-Aware Domain Retrieval)

> **Confidence-Aware Domain Retrieval for Real-Time Subtitle Systems**
>
> 모델의 불확실성(Confidence)을 기반으로 Retrieval을 동적으로 수행하고, 도메인 정보를 활용하여 검색 공간을 최소화하는 실시간 자막 시스템 연구 프로젝트.

---

# Overview

기존 Retrieval-Augmented Generation(RAG)은 대부분 입력이 들어올 때마다 Retrieval을 수행하거나, 일정한 규칙에 따라 외부 지식을 검색한다.

그러나 실시간 자막 시스템에서는 이러한 방식이 다음과 같은 문제를 가진다.

- 불필요한 Retrieval 증가
- Latency 증가
- 실시간성 저하
- 대부분의 쉬운 문장에서도 동일한 비용 발생

본 프로젝트는

> **모델이 정말 불확실한 순간에만 Retrieval을 수행한다.**

는 아이디어에서 시작하였다.

또한 검색 공간 자체를 줄이기 위해 입력의 **도메인(Domain)** 을 먼저 분석하고, 해당 도메인의 Knowledge Base만 활성화하는 구조를 제안한다.

---

# Motivation

유튜브 자막에서는 일반 문장보다

- 인터넷 밈
- 신조어
- 스포츠 용어
- 게임 용어
- 최신 AI 용어
- 고유명사

등에서 오류가 자주 발생한다.

그러나 이러한 단어는 전체 문장에서 매우 작은 비중을 차지한다.

즉,

모든 문장에서 Retrieval을 수행하는 것은 비효율적이다.

---

# Core Idea

CADR는 Retrieval을

> **필요할 때만 수행한다.**

이를 위해

1. Domain Analyzer
2. Confidence Estimator
3. Dynamic Retrieval

세 가지 모듈을 이용한다.

---

# Architecture

```text
                 Input

                   │

                   ▼

          Domain Analyzer

                   │

                   ▼

      Domain Knowledge Activation

                   │

                   ▼

            Model Inference

                   │

                   ▼

      Confidence Estimation

          │              │

          │              │

 High Confidence     Low Confidence

          │              │

          │        Dynamic Retrieval

          │              │

          └────── Logit Adjustment

                   │

                   ▼

                 Output
```

---

# Key Components

## 1. Domain Analyzer

입력으로부터

- Sports
- AI
- Game
- Music
- Cooking
- ...

등의 도메인을 추론한다.

Domain은 단일 클래스가 아닌

Multi-label 형태를 지원한다.

예)

```
Sports : 0.91

AI : 0.74

Education : 0.38
```

---

## 2. Domain Knowledge Activation

Knowledge Base는

Wikipedia

나무위키

용어집

신조어 사전

등으로 구성한다.

모든 문서는

Rule-based Domain Tagging을 통해

미리 Domain을 부여한다.

예)

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

Domain이 결정되면

전체 Knowledge Base를 검색하지 않고

선택된 Domain만 활성화한다.

---

## 3. Confidence Estimation

모델의 Softmax 결과를 이용하여

Confidence를 계산한다.

후보

- Entropy
- Top-1 Probability
- Top-1 / Top-2 Margin

등을 실험한다.

---

## 4. Dynamic Retrieval

Confidence가 충분히 높은 경우

```
Output
```

Confidence가 낮은 경우

```
Knowledge Retrieval

↓

Logit Adjustment

↓

Output
```

을 수행한다.

---

## 5. Logit Adjustment

Retrieval된 후보를 이용하여

Logit을 수정한다.

예)

초기

```
press

0.44
```

```
price

0.43
```

Football Domain

↓

```
press

+ Bias
```

↓

Softmax

↓

최종 출력

---

# Domain Lock Policy

Domain은

영상 시작 직후

고정하지 않는다.

Warm-up 구간을 거쳐

충분한 정보가 수집된 이후 Lock한다.

기본 정책

```
t ≥ W

AND

Confidence ≥ T
```

↓

Domain Lock

단,

```
Confidence ≥ T_early
```

이면

Early Lock을 수행한다.

또한

Lock 이후에는

Hysteresis를 적용하여

불필요한 Domain 변경을 방지한다.

---

# Benchmark

초기 연구는

음성을 제외하고

Text 환경에서 수행한다.

```
Sentence

↓

BERT

↓

MLM Head

↓

Confidence

↓

Retrieval

↓

Logit Adjustment
```

이를 통해

Confidence-aware Retrieval만

독립적으로 검증한다.

---

# Evaluation

## Accuracy

- Top-1 Accuracy
- Candidate Accuracy
- Domain Term Accuracy
- MRR

## Efficiency

- Retrieval Ratio
- Average Latency
- Retrieval Count
- Inference Time

## Comparison

- Baseline BERT
- Always Retrieval
- Confidence Retrieval (Ours)

---

# Candidate Scoring

전문용어는 대부분 여러 개의 Subword로 분리된다.

예)

```
LoRA

↓

Lo

##

RA
```

따라서

후보 용어의

Subword Log Probability를

합산하여

최종 Score를 계산한다.

```
Score(candidate)

=

Σ log P(subword)
```

---

# Future Work

- Whisper Decoder 적용
- Streaming ASR 적용
- Dynamic Threshold Learning
- Learned Retrieval Controller
- Knowledge Router 학습
- Domain Prior Learning
- Multi-modal Domain Analysis

---

# Research Contributions

본 연구는

기존 RAG와 달리

입력이 아니라

**모델의 내부 불확실성(Confidence)** 을 기준으로

Retrieval 여부를 결정한다.

또한

도메인을 먼저 분석하여

검색 공간을 제한한 후

필요한 경우에만 Retrieval을 수행한다.

즉,

본 프로젝트의 핵심은

> **"무엇을 검색할 것인가?"**

보다

> **"언제, 어디에서 검색할 것인가?"**

를 최적화하는 것이다.

---

# Current Status

- [x] 아이디어 설계
- [x] 시스템 아키텍처 정의
- [x] Domain Knowledge 구조 설계
- [x] Confidence 기반 Retrieval 설계
- [x] Warm-up / Lock 정책 설계
- [ ] 데이터셋 구축
- [ ] Rule-based Domain Tagging
- [ ] Knowledge Base 구축
- [ ] Domain Analyzer 구현
- [ ] Confidence Estimator 구현
- [ ] Dynamic Retrieval 구현
- [ ] Logit Adjustment 구현
- [ ] BERT 기반 Benchmark
- [ ] Whisper 적용