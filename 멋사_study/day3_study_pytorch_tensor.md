# 3일차 - 스터디

## 파이토치 기초 - 텐서 조작하기

## 1. 벡터, 행렬 그리고 텐서

### 1) 텐서 이해하기

| 차원 | 이름 |
|------|------|
| 0차원 상수 | 스칼라 (Scalar) |
| 1차원 텐서 | 벡터 (Vector) |
| 2차원 텐서 | 행렬 (Matrix) |
| 3차원 이상 | 텐서 (Tensor) |

### 2) 파이토치 텐서

- 행렬의 가로축은 `batch_size`, 세로축은 `dim`(차원).
- 하나의 데이터 크기가 바로 벡터의 길이가 된다.
- `dim`은 벡터의 길이. 즉, **데이터 하나의 길이 = 벡터의 길이 = dim**.
- 컴퓨터가 한 번에 읽어들이는 덩어리가 바로 `batch_size`가 된다.

**3차원에서는**

- 자연어 : `(batch_size, 문장 길이, dim(단어당 벡터 길이))`
- 이미지 : `(batch_size, 가로, 세로)`

쉽게 분석해보자면, 하나의 데이터는 **하나의 행마다 단어의 벡터를 열만큼 나열한 행렬**이 된다.
그리고 데이터를 여러 개 붙이면? `batch_size`가 늘어나는 것.

---

## 2. 파이토치로 텐서 만들기

```python
import torch
```

> - `-1`은 최대 차원. `-2`는 최대 차원 밑의 차원.
> - 슬라이싱 기호 → `:`

### 1) 1D

```python
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱
```

### 2) 2D

```python
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]])

print(t.dim())   # rank. 즉, 차원
print(t.size())  # shape
# 2
# torch.Size([4, 3])

print(t[:, 1])          # 전체 행 선택 후 두 번째 열만 가져온다
print(t[:, 1].size())   # ↑ 위 경우의 크기
# tensor([ 2.,  5.,  8., 11.])
# torch.Size([4])
```

> `dim`으로 입력하는 것은 차원을 뜻함 (벡터 = 1차원, 행렬 = 2차원 등).
> 앞서 말한 **벡터의 길이로서의 dim**과는 다르다는 것 유의할 것.

### 3) 브로드캐스팅

행렬의 크기가 맞지 않을 때 계산을 수행하면 프로그램이 임의로 행렬 크기를 맞춰서 계산한다.

```python
# 브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되는지
[1, 2]
==> [[1, 2],
     [1, 2]]

[3]
[4]
==> [[3, 3],
     [4, 4]]
```

> 편리하지만 의도치 않은 크기 자동 변환으로 버그를 만들 수 있으니 주의.

---

## 3. 파이토치 다른 연산

### 1) 행렬 곱 vs 그냥 곱

```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print(m1.matmul(m2))  # 행렬 곱 (2x2) x (2x1) = (2x1)
print(m1.mul(m2))     # 그냥 곱 (원소별 곱). 크기 다르면 브로드캐스팅
print(m1 * m2)        # mul과 동일
```

- `matmul()` : 행렬 곱
- `mul()` (또는 `*`) : 원소별(element-wise) 곱, 크기 다르면 브로드캐스팅

### 2) 평균 (mean)

```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.mean())          # 전체 평균
print(t.mean(dim=0))     # 행을 제거 → 열 방향 평균: [2., 3.]
print(t.mean(dim=1))     # 열을 제거 → 행 방향 평균: [1.5, 3.5]
```

### 3) 합 (sum)

```python
print(t.sum())        # 전체 합
print(t.sum(dim=0))   # 행 제거 → 열 방향 합
print(t.sum(dim=1))   # 열 제거 → 행 방향 합
```

> **dim 직관:** `dim=i`는 "i번째 차원을 제거(축소)한다"는 의미.
> `dim=0`이면 행(첫 번째 차원)이 사라지고 열 방향으로 연산이 합쳐진다.

### 4) 최댓값 (max) & 최대 인덱스 (argmax)

```python
t = torch.FloatTensor([[1, 2], [3, 4]])

print(t.max())          # 전체 최댓값 하나: tensor(4.)

# dim을 주면 (max값, argmax인덱스) 튜플을 반환
print(t.max(dim=0))
# (tensor([3., 4.]), tensor([1, 1]))

print(t.max(dim=0)[0])  # max 값만
print(t.max(dim=0)[1])  # argmax 인덱스만

print(t.argmax(dim=0))  # 최댓값의 인덱스만 바로 반환: tensor([1, 1])
```

---

## 4. 기타 연산들

### 1) View (Reshape) — 원소 수를 유지하며 모양 변경

가장 많이 쓰는 연산. 텐서 안의 원소 개수는 그대로 두고 shape만 바꾼다.

```python
t = torch.FloatTensor([[[0, 1, 2], [3, 4, 5]],
                       [[6, 7, 8], [9, 10, 11]]])
print(t.shape)  # torch.Size([2, 2, 3])

print(t.view([-1, 3]))         # (4, 3)으로 변경
print(t.view([-1, 3]).shape)   # torch.Size([4, 3])

print(t.view([-1, 1, 3]).shape)  # torch.Size([4, 1, 3])
```

> `-1`은 "나머지는 알아서 계산하라"는 의미. 전체 원소 수가 유지되도록 자동 결정된다.
> 위 예시: 총 원소 12개 → `(-1, 3)`이면 12/3 = 4 → `(4, 3)`.

### 2) Squeeze — 크기가 1인 차원 제거

```python
t = torch.FloatTensor([[0], [1], [2]])
print(t.shape)            # torch.Size([3, 1])

print(t.squeeze())        # tensor([0., 1., 2.])
print(t.squeeze().shape)  # torch.Size([3])
```

> `squeeze(dim=i)`로 특정 차원만 지정해 제거할 수도 있다. 해당 차원 크기가 1일 때만 제거됨.

### 3) Unsqueeze — 특정 위치에 크기 1인 차원 추가

```python
t = torch.FloatTensor([0, 1, 2])
print(t.shape)               # torch.Size([3])

print(t.unsqueeze(0))        # tensor([[0., 1., 2.]])
print(t.unsqueeze(0).shape)  # torch.Size([1, 3])

print(t.unsqueeze(1).shape)  # torch.Size([3, 1])
```

> `unsqueeze`는 `squeeze`의 반대. `view`로도 동일하게 구현 가능
> (예: `t.view(1, -1)` == `t.unsqueeze(0)`).

### 4) Type Casting — 자료형 변환

```python
lt = torch.LongTensor([1, 2, 3, 4])
print(lt.float())   # tensor([1., 2., 3., 4.])  → FloatTensor

ft = torch.FloatTensor([1.5, 2.5])
print(ft.long())    # tensor([1, 2])  → LongTensor (소수점 버림)

bt = torch.ByteTensor([True, False, True])
print(bt.long())    # 불리언 → 정수
print(bt.float())   # 불리언 → 실수
```

> 비교 연산(`t > 0` 등)의 결과는 ByteTensor(불리언)로 나오는데,
> 이를 `.float()`이나 `.long()`으로 캐스팅해서 자주 활용한다.

### 5) Concatenate — 텐서 이어 붙이기 (`torch.cat`)

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))  # (4, 2) — 세로로 연결
print(torch.cat([x, y], dim=1))  # (2, 4) — 가로로 연결
```

> 어느 차원으로 늘릴지를 `dim`으로 지정. 합치는 차원 외의 크기는 같아야 한다.

### 6) Stacking — 텐서 쌓기 (`torch.stack`)

```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))         # (3, 2)
print(torch.stack([x, y, z], dim=1))  # (2, 3)
```

> `cat`은 기존 차원을 따라 이어 붙이지만, `stack`은 **새로운 차원을 만들어** 쌓는다.
> `torch.stack([x, y, z])` ≈ `torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0)`.

### 7) ones_like / zeros_like — 같은 모양의 1 또는 0 텐서

```python
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])

print(torch.ones_like(x))   # x와 같은 shape의 1로 채운 텐서
print(torch.zeros_like(x))  # x와 같은 shape의 0으로 채운 텐서
```

> 같은 디바이스(CPU/GPU), 같은 자료형으로 생성된다는 점이 장점.

### 8) In-place Operation — 덮어쓰기 연산

연산 결과를 새 텐서가 아니라 기존 텐서에 바로 덮어쓴다. 함수명 뒤에 `_`(언더스코어)를 붙인다.

```python
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))  # 결과만 반환, x는 그대로
print(x)          # 변화 없음

print(x.mul_(2.)) # x 자체를 덮어씀
print(x)          # 값이 변경됨
```

> 메모리는 절약되지만, 자동 미분(autograd) 과정에서 문제를 일으킬 수 있어
> 학습 코드에서는 주의해서 사용한다.

---

## 정리

- **shape 조작:** `view`, `squeeze`, `unsqueeze`
- **합치기:** `cat`(기존 차원), `stack`(새 차원)
- **자료형:** `.float()`, `.long()`, `.byte()`
- **편의 생성:** `ones_like`, `zeros_like`
- **덮어쓰기:** `_` 접미사 연산 (`mul_`, `add_` 등)
- **dim 핵심 직관:** `dim=i`는 "i번째 차원을 축소/기준으로 삼는다"
