# CIFAR-10 Improved CNN Classifier

PyTorch를 사용하여 CIFAR-10 이미지 데이터를 분류하는 개선된 CNN 프로젝트입니다.

이 프로젝트는 이전 `CIFAR-10 Basic CNN` 프로젝트를 기반으로, 모델 성능을 개선하기 위해 Data Augmentation, Batch Normalization, Dropout을 추가한 실험입니다.

## Project Goal

기본 CNN 모델에 몇 가지 개선 기법을 추가하여 CIFAR-10 분류 성능이 어떻게 변하는지 확인합니다.

추가한 개선 요소는 다음과 같습니다.

- Data Augmentation
- BatchNorm2d
- Dropout
- Deeper CNN 구조

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

## Transform

이번 프로젝트에서는 학습 데이터와 테스트 데이터에 서로 다른 transform을 적용했습니다.

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Transform 설명

- `RandomHorizontalFlip()`  
  학습 이미지를 랜덤하게 좌우 반전합니다.

- `RandomCrop(32, padding=4)`  
  이미지 주변에 padding을 추가한 뒤 랜덤하게 잘라냅니다.

- `ToTensor()`  
  이미지를 PyTorch Tensor로 변환합니다.

- `Normalize()`  
  이미지 값을 정규화하여 학습을 더 안정적으로 만듭니다.

학습 데이터에는 Data Augmentation을 적용하고, 테스트 데이터에는 평가의 일관성을 위해 랜덤 변형을 적용하지 않았습니다.

## Model Structure

이번 프로젝트에서는 기본 CNN보다 더 깊은 CNN 구조를 사용했습니다.

```text
Input Image: 3 × 32 × 32

→ Conv2d: 3 → 32
→ BatchNorm2d
→ ReLU
→ MaxPool2d

→ Conv2d: 32 → 64
→ BatchNorm2d
→ ReLU
→ MaxPool2d

→ Conv2d: 64 → 128
→ BatchNorm2d
→ ReLU
→ MaxPool2d

→ Flatten
→ Linear: 128 × 4 × 4 → 256
→ ReLU
→ Dropout
→ Linear: 256 → 10
```

마지막 출력값 10개는 CIFAR-10의 10개 클래스에 대한 예측 점수를 의미합니다.

## CNN Shape Flow

```text
Input: [3, 32, 32]

Conv + Pool
→ [32, 16, 16]

Conv + Pool
→ [64, 8, 8]

Conv + Pool
→ [128, 4, 4]

Flatten
→ 128 × 4 × 4 = 2048

Linear
→ 256

Linear
→ 10
```

## Training Settings

| Setting | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 5e-4 |
| Batch Size | 64 |
| Epochs | 10 |
| Device | CPU |

## Experiment Results

| Model | Epochs | Optimizer | Learning Rate | Accuracy | Avg Loss |
|---|---:|---|---:|---:|---:|
| Basic CNN | 10 | Adam | 5e-4 | 71.7% | 1.054057 |
| Improved CNN | 10 | Adam | 5e-4 | 75.3% | 0.703456 |

## Result Analysis

기본 CNN 모델의 정확도는 71.7%였고, Avg Loss는 1.054057이었습니다.

이후 Data Augmentation, BatchNorm2d, Dropout을 추가한 Improved CNN 모델에서는 정확도가 75.3%로 상승했고, Avg Loss는 0.703456으로 감소했습니다.

이를 통해 데이터 증강과 모델 구조 개선이 CIFAR-10 분류 성능 향상에 도움이 된다는 것을 확인했습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- Data Augmentation의 기본 개념
- `RandomHorizontalFlip` 사용법
- `RandomCrop` 사용법
- 학습용 transform과 테스트용 transform을 분리하는 이유
- `BatchNorm2d`를 이용한 학습 안정화
- `Dropout`을 이용한 과적합 완화
- CNN layer를 추가했을 때 feature map 크기가 변하는 과정
- Basic CNN과 Improved CNN의 성능 비교
- optimizer와 learning rate 설정의 중요성

## Key Concepts

### Data Augmentation

Data Augmentation은 학습 이미지를 랜덤하게 변형하여 데이터가 더 다양해진 것처럼 학습시키는 방법입니다.

이를 통해 모델이 특정 이미지 형태에만 과하게 맞춰지는 것을 줄이고, 일반화 성능을 높일 수 있습니다.

### BatchNorm2d

`BatchNorm2d`는 CNN의 feature map 값을 정규화하여 학습을 더 안정적으로 만들어주는 층입니다.

### Dropout

`Dropout`은 학습 중 일부 뉴런을 랜덤하게 꺼서 모델이 특정 뉴런에만 의존하지 않도록 만드는 방법입니다.

과적합을 줄이는 데 도움이 됩니다.

### Feature Map

Feature map은 `Conv2d`를 통과한 뒤 만들어지는 특징 결과물입니다.

CNN은 원본 이미지를 바로 분류하는 것이 아니라, 여러 feature map을 통해 이미지의 선, 색 변화, 무늬, 윤곽 같은 특징을 추출한 뒤 분류합니다.

## Limitations

이번 모델은 기본 CNN보다 성능이 개선되었지만, 여전히 단순한 CNN 구조입니다.

CIFAR-10은 실제 사물 이미지로 구성되어 있어 클래스 간 형태가 비슷한 경우가 많습니다. 예를 들어 `cat`과 `dog`, `deer`와 `horse`, `automobile`과 `truck`은 기본 CNN 구조에서 혼동될 수 있습니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- 더 깊은 CNN 구조
- Learning Rate Scheduler
- 더 다양한 Data Augmentation
- Residual Block
- ResNet 구조 적용

## Next Step

다음 단계에서는 대표적인 CNN 구조를 직접 구현해볼 예정입니다.

후보:

- VGG-style CNN
- ResNet BasicBlock
- 논문 구조 따라 구현하기

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/intro.html
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
