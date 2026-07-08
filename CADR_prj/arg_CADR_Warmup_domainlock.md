# arg_CADR

## 추가 논의: Warm-up 및 Domain Lock 정책

### 논의 배경

실시간 자막 시스템에서는 도메인을 가능한 빨리 결정해야 하지만, 너무 이른
판단은 오분류를 유발할 수 있다. 따라서 실시간성과 안정성 사이의 균형을
맞추는 정책이 필요하다.

### 1. Warm-up 기준

-   문장 수 기준도 고려했으나, 자막은 문장 경계가 일정하지 않아
    시스템마다 결과가 달라질 수 있다.
-   최종적으로 **토큰(Token) 수 기준**을 채택한다.
-   토큰은 모델 내부의 일관된 단위이며 구현도 단순하다.

### 2. Domain Lock 정책

기본 정책: - Warm-up 종료 (`t ≥ W`) - Confidence ≥ T

두 조건을 모두 만족하면 Domain Lock을 수행한다.

### 3. Early Lock

Confidence가 매우 높은 경우(`Confidence ≥ T_early`, `T_early > T`)에는
Warm-up 종료를 기다리지 않고 즉시 Lock하는 예외 정책을 적용한다.

### 4. Domain 유지 정책

한 번 Lock된 도메인은 작은 Confidence 변화만으로 즉시 변경하지 않는다.
충분한 반대 증거가 누적될 때만 Unlock 또는 재분류를 수행하는 Hysteresis
정책을 적용한다.

## 결론

-   Warm-up: 최근 N개의 토큰 기준
-   기본 Lock: `t ≥ W` AND `Confidence ≥ T`
-   예외: `Confidence ≥ T_early`이면 Early Lock
-   Domain 변경: Hysteresis 적용
-   향후 Warm-up 길이와 Threshold는 실험적으로 최적화한다.
