# PyTorch 03 - Transform

## 1. Transform이란?

데이터가 항상 모델 학습에 바로 사용할 수 있는 형태로 제공되지는 않는다.  
그래서 `transform`을 사용해 데이터를 학습에 적합한 형태로 변환한다.

PyTorch에서는 주로 두 가지 변형을 사용한다.

```text
transform
→ 이미지 feature에 적용하는 변형

target_transform
→ 정답 label에 적용하는 변형
```

---

## 2. FashionMNIST에 Transform 적용하기

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            0, torch.tensor(y), value=1
        )
    )
)
```

### 코드 의미

```python
transform=ToTensor()
```

이미지 데이터를 PyTorch Tensor로 변환한다.

```python
target_transform=Lambda(...)
```

정답 label을 변환한다.

---

## 3. ToTensor()

`ToTensor()`는 이미지를 PyTorch Tensor로 변환하는 함수이다.

PIL Image나 NumPy 배열을 PyTorch `FloatTensor`로 변환한다.  
또한 이미지 픽셀 값을 `[0, 255]` 범위에서 `[0.0, 1.0]` 범위로 조정한다.

```text
PIL Image / NumPy 배열
→ PyTorch Tensor
```

즉, 모델 학습에 사용할 수 있는 Tensor 형태로 이미지를 바꾸는 역할을 한다.

---

## 4. Lambda

`Lambda`는 내가 직접 만든 간단한 변환 함수를 적용할 때 사용한다.

위 코드에서는 정수 label을 원-핫 인코딩된 Tensor로 변환한다.

```python
target_transform = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(
        0, torch.tensor(y), value=1
    )
)
```

---

## 5. 원-핫 인코딩

FashionMNIST의 label은 원래 정수 하나이다.

예를 들어:

```text
label = 9
```

하지만 위 코드에서는 label을 원-핫 인코딩 형태로 바꾼다.

```text
label = 9
→ [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

원-핫 인코딩은 정답 위치만 `1`이고, 나머지는 `0`인 벡터로 표현하는 방식이다.

FashionMNIST는 클래스가 10개이므로 길이가 10인 벡터를 사용한다.

---

## 6. 핵심 코드

```python
torch.zeros(10, dtype=torch.float)
```

길이가 10인 0 벡터를 만든다.

```python
scatter_(0, torch.tensor(y), value=1)
```

label 번호에 해당하는 위치에 `1`을 넣는다.

예를 들어 `y = 3`이면:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

---

## 정리

- `Transform`은 데이터를 모델 학습에 적합한 형태로 바꾸는 과정이다.
- `transform`은 이미지 feature에 적용한다.
- `target_transform`은 정답 label에 적용한다.
- `ToTensor()`는 이미지를 PyTorch Tensor로 변환한다.
- `ToTensor()`는 픽셀 값을 `[0, 255]`에서 `[0.0, 1.0]` 범위로 조정한다.
- `Lambda`는 사용자 정의 변환 함수를 적용할 때 사용한다.
- 원-핫 인코딩은 정답 label 위치만 `1`이고 나머지는 `0`인 벡터로 표현하는 방식이다.

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/transforms_tutorial.html
