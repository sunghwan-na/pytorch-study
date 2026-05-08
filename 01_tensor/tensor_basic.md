# PyTorch 01 - Tensor

## 1. Tensor란?

Tensor는 PyTorch에서 데이터를 표현하는 기본 자료구조이다.  
배열이나 행렬과 비슷하며, 모델의 입력, 출력, 매개변수 등을 표현할 때 사용된다.

Tensor는 NumPy 배열과 유사하지만, GPU에서 연산할 수 있고 자동 미분에 최적화되어 있다는 특징이 있다.

```python
import torch
import numpy as np
```

---

## 2. Tensor 초기화

Tensor는 여러 가지 방법으로 생성할 수 있다.

### 2-1. 데이터로부터 직접 생성하기

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

`torch.tensor(data)`는 파이썬 리스트 같은 데이터를 Tensor로 변환하는 함수이다.  
데이터의 자료형은 자동으로 유추된다.

---

### 2-2. NumPy 배열로부터 생성하기

```python
data = [[1, 2], [3, 4]]

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(np_array)
print(x_np)
```

`torch.from_numpy()`는 NumPy 배열을 PyTorch Tensor로 변환하는 함수이다.

출력 예시:

```python
[[1 2]
 [3 4]]

tensor([[1, 2],
        [3, 4]])
```

---

### 2-3. 다른 Tensor로부터 생성하기

기존 Tensor를 기준으로 새로운 Tensor를 만들 수 있다.  
명시적으로 자료형을 바꾸지 않으면 기존 Tensor의 `shape`와 `dtype`을 유지한다.

```python
x_data = torch.tensor([[1, 2], [3, 4]])

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor:\n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")
```

- `torch.ones_like(x_data)`: `x_data`와 같은 모양의 Tensor를 만들고 전부 1로 채움
- `torch.rand_like(x_data, dtype=torch.float)`: `x_data`와 같은 모양의 랜덤 Tensor를 만들고 자료형은 실수형으로 지정

---

### 2-4. 무작위 또는 상수 값으로 생성하기

`shape`는 Tensor의 각 차원 크기를 나타내는 튜플이다.

```python
shape = (2, 3)

rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}")
```

- `torch.rand(shape)`: 0과 1 사이의 랜덤값으로 채운 Tensor 생성
- `torch.ones(shape)`: 1로 채운 Tensor 생성
- `torch.zeros(shape)`: 0으로 채운 Tensor 생성

예를 들어 `shape = (2, 3)`은 2행 3열 Tensor를 의미한다.

---

## 3. Tensor의 속성

Tensor의 속성은 Tensor의 모양, 자료형, 저장 장치를 나타낸다.

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

- `shape`: Tensor의 모양
- `dtype`: Tensor의 자료형
- `device`: Tensor가 저장된 장치

출력 예시:

```python
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---

## 4. Tensor 연산

Tensor는 인덱싱, 슬라이싱, 전치, 수학 계산, 선형 대수 연산 등 다양한 연산을 지원한다.

### 4-1. 인덱싱과 슬라이싱

```python
tensor = torch.ones(4, 4)

print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:, 1] = 0
print(tensor)
```

- `tensor[0]`: 첫 번째 행
- `tensor[:, 0]`: 모든 행의 첫 번째 열
- `tensor[..., -1]`: 마지막 열
- `tensor[:, 1] = 0`: 모든 행의 두 번째 열을 0으로 변경

---

### 4-2. Tensor 합치기

```python
tensor = torch.ones(4, 4)
t1 = torch.cat([tensor, tensor, tensor], dim=1)

print(t1)
```

`torch.cat()`은 여러 Tensor를 특정 차원 방향으로 이어 붙이는 함수이다.

2차원 Tensor 기준:

- `dim=0`: 위아래로 붙임, 행 개수가 늘어남
- `dim=1`: 옆으로 붙임, 열 개수가 늘어남

예를 들어 `(4, 4)` Tensor 3개를 `dim=1`로 연결하면 결과는 `(4, 12)`가 된다.

`torch.stack()`은 기존 차원으로 이어 붙이는 것이 아니라 새로운 차원을 만들어 쌓는 방식이다.

---

## 5. 산술 연산

### 5-1. 행렬 곱

```python
tensor = torch.ones(4, 4)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
```

- `@`: 행렬 곱
- `matmul()`: 행렬 곱
- `tensor.T`: Tensor 전치
- `out=y3`: 계산 결과를 `y3`에 저장

`y1`, `y2`, `y3`는 모두 같은 값을 가진다.

---

### 5-2. 요소별 곱

```python
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

- `*`: 요소별 곱
- `mul()`: 요소별 곱
- `out=z3`: 계산 결과를 `z3`에 저장

요소별 곱은 같은 위치에 있는 값끼리 곱하는 연산이다.

---

## 6. 단일 요소 Tensor

Tensor의 모든 값을 하나로 집계하면 요소가 하나인 Tensor가 된다.  
이때 `item()`을 사용하면 Python 숫자 값으로 변환할 수 있다.

```python
tensor = torch.ones(4, 4)

agg = tensor.sum()
agg_item = agg.item()

print(agg_item, type(agg_item))
```

- `tensor.sum()`: Tensor 안의 모든 값을 더해 단일 값 Tensor로 만듦
- `item()`: 요소가 하나인 Tensor를 Python 숫자로 변환

---

## 7. In-place 연산

In-place 연산은 연산 결과를 새로운 Tensor에 저장하지 않고 기존 Tensor 자체를 직접 수정하는 연산이다.  
PyTorch에서는 함수 이름 뒤에 `_`가 붙는다.

```python
tensor.add_(5)
```

`tensor.add_(5)`는 Tensor의 모든 값에 5를 더하고, 기존 Tensor 자체를 수정한다.

In-place 연산은 메모리를 절약할 수 있지만, 자동 미분 과정에서 문제가 생길 수 있으므로 초반에는 사용을 권장하지 않는다.

---

## 8. NumPy 변환

CPU 상의 Tensor와 NumPy 배열은 메모리를 공유할 수 있다.  
따라서 한쪽을 변경하면 다른 한쪽도 변경될 수 있다.

### 8-1. Tensor를 NumPy 배열로 변환

```python
t = torch.ones(5)
n = t.numpy()
```

- `t.numpy()`: Tensor를 NumPy 배열로 변환

---

### 8-2. NumPy 배열을 Tensor로 변환

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

- `torch.from_numpy(n)`: NumPy 배열을 Tensor로 변환

---

## 정리

- Tensor는 PyTorch의 기본 자료구조이다.
- `torch.tensor()`로 데이터를 직접 Tensor로 만들 수 있다.
- `torch.from_numpy()`로 NumPy 배열을 Tensor로 변환할 수 있다.
- `shape`, `dtype`, `device`는 Tensor의 중요한 속성이다.
- `@`, `matmul()`은 행렬 곱이다.
- `*`, `mul()`은 요소별 곱이다.
- `item()`은 단일 값 Tensor를 Python 숫자로 변환한다.
- `_`가 붙은 함수는 기존 Tensor를 직접 수정하는 in-place 연산이다.

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html
