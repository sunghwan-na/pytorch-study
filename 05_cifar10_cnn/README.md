# CIFAR-10 CNN Classifier

PyTorch를 사용하여 CIFAR-10 이미지 데이터를 분류하는 CNN 프로젝트입니다.

이 프로젝트의 목적은 FashionMNIST MLP 프로젝트에서 익힌 PyTorch 학습 흐름을 바탕으로, CNN 구조를 사용해 컬러 이미지 분류를 경험하는 것입니다.

## Project Goal

CIFAR-10 이미지를 입력받아 10개의 클래스 중 하나로 분류하는 CNN 모델을 학습합니다.

## Dataset

사용한 데이터셋은 CIFAR-10입니다.

- 학습 데이터: 50,000개
- 테스트 데이터: 10,000개
- 이미지 크기: 32 × 32
- 이미지 형태: RGB 컬러 이미지
- 클래스 수: 10개

클래스는 다음과 같습니다.

```text
0: airplane
1: automobile
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
```

## Difference from FashionMNIST

이전 FashionMNIST 프로젝트와 비교하면 CIFAR-10은 더 복잡한 데이터셋입니다.

```text
FashionMNIST
→ 흑백 이미지
→ 1 × 28 × 28
→ 의류 이미지

CIFAR-10
→ 컬러 이미지
→ 3 × 32 × 32
→ 실제 사물 이미지
```

따라서 CIFAR-10은 FashionMNIST보다 분류 난이도가 더 높습니다.

## Transform

CIFAR-10 이미지를 모델 학습에 사용할 수 있도록 Tensor로 변환하고 정규화를 적용했습니다.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Transform 의미

- `ToTensor()`  
  이미지를 PyTorch Tensor로 변환합니다.

- `Normalize()`  
  이미지 값을 정규화하여 학습을 더 안정적으로 만듭니다.

CIFAR-10 이미지 하나의 shape는 다음과 같습니다.

```text
[3, 32, 32]
```

의미:

```text
3  → RGB 채널
32 → 이미지 세로
32 → 이미지 가로
```

## Model Structure

이번 프로젝트에서는 기본 CNN 모델을 사용했습니다.

모델 구조는 다음과 같습니다.

```text
Input Image: 3 × 32 × 32
→ Conv2d: 3 → 32
→ ReLU
→ MaxPool2d
→ Conv2d: 32 → 64
→ ReLU
→ MaxPool2d
→ Flatten
→ Linear: 64 × 8 × 8 → 128
→ ReLU
→ Linear: 128 → 10
```

마지막 출력값 10개는 CIFAR-10의 10개 클래스를 의미합니다.

## CNN Shape Flow

이미지 크기 변화는 다음과 같습니다.

```text
Input: [3, 32, 32]

Conv2d + MaxPool2d
→ [32, 16, 16]

Conv2d + MaxPool2d
→ [64, 8, 8]

Flatten
→ 64 × 8 × 8 = 4096

Linear
→ 128

Linear
→ 10
```

## Training Settings

| Setting | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 10 |
| Device | CPU |

## Experiment Results

| Model | Epochs | Optimizer | Learning Rate | Accuracy | Avg Loss |
|---|---:|---|---:|---:|---:|
| Basic CNN | 10 | SGD | 1e-2 | 63.1% | 1.040675 |
| Basic CNN | 10 | Adam | 1e-3 | 71.3% | 1.757582 |
| Basic CNN | 10 | Adam | 5e-4 | 71.7% | 1.054057 |

## Result Analysis

SGD optimizer를 사용했을 때 정확도는 63.1%였습니다.

Adam optimizer로 변경하자 정확도는 71.3%까지 상승했습니다.  
하지만 learning rate가 `1e-3`일 때는 평균 loss가 높게 나타났습니다.

이후 Adam을 유지하고 learning rate를 `5e-4`로 낮추자 정확도는 71.7%로 소폭 상승했고, 평균 loss도 1.054057로 감소했습니다.

이를 통해 optimizer와 learning rate 설정이 모델 성능과 학습 안정성에 큰 영향을 줄 수 있음을 확인했습니다.

## Prediction Visualization

학습된 모델을 사용하여 테스트 이미지 9개를 랜덤으로 선택하고, 모델의 예측값과 실제 정답을 비교했습니다.

예측 결과를 시각화한 결과, `airplane`, `frog`, `truck` 등은 비교적 잘 분류했습니다.

반면 `deer`, `dog`, `horse`처럼 형태가 비슷한 동물 클래스에서는 오분류가 발생했습니다.

예시:

```text
Pred: horse
True: deer

Pred: bird
True: deer

Pred: horse
True: dog
```

CIFAR-10 이미지는 32×32 크기의 저해상도 컬러 이미지이기 때문에 비슷한 형태의 클래스는 기본 CNN 모델에서 구분하기 어려울 수 있습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- CIFAR-10 데이터셋 불러오기
- 컬러 이미지 Tensor shape 이해하기
- `[3, 32, 32]` 이미지 구조 이해하기
- CNN 모델 구조 작성하기
- `Conv2d`, `ReLU`, `MaxPool2d` 사용하기
- Feature Map 개념 이해하기
- `Flatten` 후 Linear layer로 분류하기
- `CrossEntropyLoss`를 이용한 분류 손실 계산
- SGD와 Adam optimizer 비교하기
- learning rate 변경에 따른 성능 차이 확인하기
- 예측 결과 시각화하기

## Key Training Flow

```text
데이터 불러오기
→ Transform 적용
→ DataLoader 생성
→ CNN 모델 정의
→ loss 함수 설정
→ optimizer 설정
→ 학습
→ 평가
→ 모델 저장
→ 예측 결과 시각화
```

## Limitations

이번 모델은 기본적인 CNN 구조만 사용했습니다.

CIFAR-10은 FashionMNIST보다 복잡한 컬러 이미지 데이터셋이기 때문에, 단순한 CNN 구조만으로는 높은 정확도를 얻는 데 한계가 있습니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- CNN layer 추가
- Batch Normalization 사용
- Dropout 사용
- Data Augmentation 적용
- 더 깊은 모델 구조 사용
- ResNet 같은 대표 CNN 구조 학습

## Next Step

다음 단계에서는 CNN 구조를 개선하여 CIFAR-10 분류 성능을 높이는 실험을 진행할 예정입니다.

추가로 학습할 개념은 다음과 같습니다.

- Data Augmentation
- Batch Normalization
- Dropout
- Deeper CNN
- ResNet 구조

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/intro.html
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
