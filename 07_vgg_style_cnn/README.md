# VGG-style CNN Classifier

PyTorch를 사용하여 CIFAR-10 이미지 데이터를 분류하는 VGG-style CNN 프로젝트입니다.

이 프로젝트는 이전 `CIFAR-10 Basic CNN`, `CIFAR-10 Improved CNN` 프로젝트를 기반으로, Conv layer를 더 깊게 쌓았을 때 성능이 어떻게 변하는지 확인하는 실험입니다.

## Project Goal

CIFAR-10 데이터셋을 VGG-style CNN 구조로 학습하여 이미지 분류 성능을 개선합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- VGG-style CNN 구조 이해하기
- Conv layer를 깊게 쌓는 방식 익히기
- CNN의 Shape Flow 이해하기
- Basic CNN / Improved CNN / VGG-style CNN 성능 비교하기

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

학습 데이터와 테스트 데이터에 서로 다른 transform을 적용했습니다.

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

이번 프로젝트에서는 VGG-style CNN 구조를 사용했습니다.

기본 CNN은 보통 다음과 같은 구조입니다.

```text
Conv → Pool
Conv → Pool
Conv → Pool
```

VGG-style CNN은 Pooling 전에 Conv layer를 여러 번 쌓습니다.

```text
Conv → Conv → Pool
Conv → Conv → Pool
Conv → Conv → Pool
```

Pooling으로 이미지 크기를 줄이기 전에 Conv layer를 여러 번 사용하여 더 많은 특징을 추출하는 구조입니다.

## VGG-style CNN Architecture

```text
Input Image: 3 × 32 × 32

Block 1
→ Conv2d: 3 → 32
→ BatchNorm2d
→ ReLU
→ Conv2d: 32 → 32
→ BatchNorm2d
→ ReLU
→ MaxPool2d

Block 2
→ Conv2d: 32 → 64
→ BatchNorm2d
→ ReLU
→ Conv2d: 64 → 64
→ BatchNorm2d
→ ReLU
→ MaxPool2d

Block 3
→ Conv2d: 64 → 128
→ BatchNorm2d
→ ReLU
→ Conv2d: 128 → 128
→ BatchNorm2d
→ ReLU
→ MaxPool2d

Classifier
→ Flatten
→ Linear: 128 × 4 × 4 → 256
→ ReLU
→ Dropout
→ Linear: 256 → 10
```

마지막 출력값 10개는 CIFAR-10의 10개 클래스에 대한 예측 점수를 의미합니다.

## Shape Flow

```text
Input: [3, 32, 32]

Block 1
Conv 3 → 32
Conv 32 → 32
MaxPool
→ [32, 16, 16]

Block 2
Conv 32 → 64
Conv 64 → 64
MaxPool
→ [64, 8, 8]

Block 3
Conv 64 → 128
Conv 128 → 128
MaxPool
→ [128, 4, 4]

Flatten
→ 128 × 4 × 4 = 2048

Linear
→ 256

Linear
→ 10
```

중요한 점은 다음과 같습니다.

```text
Conv2d
→ feature map 개수를 바꾼다.

MaxPool2d
→ 이미지의 공간 크기를 줄인다.

Flatten
→ CNN이 추출한 feature map을 Linear layer에 넣기 위해 1차원으로 펼친다.
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
| VGG-style CNN | 10 | Adam | 5e-4 | 81.4% | 0.549206 |

## Result Analysis

Basic CNN의 정확도는 71.7%, Improved CNN의 정확도는 75.3%였습니다.

VGG-style CNN에서는 Conv layer를 더 깊게 쌓아 특징 추출 능력을 강화했고, 그 결과 정확도가 81.4%까지 상승했습니다.

또한 Avg Loss도 0.549206으로 감소하여 이전 모델들보다 예측과 정답의 차이가 줄어든 것을 확인했습니다.

이를 통해 CIFAR-10 이미지 분류에서 단순한 CNN보다 깊은 CNN 구조가 더 좋은 성능을 낼 수 있음을 확인했습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- VGG-style CNN 구조 이해
- Conv layer를 여러 개 쌓는 방식
- Conv2d가 feature map을 만드는 과정
- MaxPool2d를 통한 이미지 크기 감소
- BatchNorm2d를 이용한 학습 안정화
- Dropout을 이용한 과적합 완화
- Shape Flow 계산
- Flatten 이후 Linear layer로 분류하는 흐름
- Basic CNN, Improved CNN, VGG-style CNN 성능 비교

## Key Concepts

### VGG-style CNN

VGG-style CNN은 Conv layer를 여러 개 연속으로 쌓고, 이후 MaxPool2d를 사용해 이미지 크기를 줄이는 구조입니다.

이 방식은 Pooling 전에 더 많은 특징을 추출할 수 있다는 장점이 있습니다.

### Feature Map

Feature map은 Conv2d를 통과한 뒤 만들어지는 특징 결과물입니다.

CNN은 원본 이미지를 바로 분류하는 것이 아니라, 여러 feature map을 통해 선, 색 변화, 무늬, 윤곽, 물체 형태 같은 특징을 추출한 뒤 분류합니다.

### BatchNorm2d

BatchNorm2d는 CNN 중간의 feature map 값을 정규화하여 학습을 안정적으로 만들어주는 층입니다.

### Dropout

Dropout은 학습 중 일부 뉴런을 랜덤하게 꺼서 모델이 특정 뉴런이나 패턴에만 의존하지 않도록 만드는 방법입니다.

과적합을 줄이는 데 도움이 됩니다.

## Prediction Visualization

학습된 모델을 사용하여 테스트 이미지 9개를 랜덤으로 선택하고, 모델의 예측값과 실제 정답을 비교했습니다.

예측 결과는 다음과 같은 형태로 시각화했습니다.

```text
Pred: frog
True: frog

Pred: truck
True: truck

Pred: deer
True: horse
```

`Pred`는 모델의 예측값이고, `True`는 실제 정답입니다.

CIFAR-10은 32×32 크기의 저해상도 이미지이기 때문에 `cat`과 `dog`, `deer`와 `horse`, `automobile`과 `truck`처럼 형태가 비슷한 클래스는 혼동될 수 있습니다.

## Limitations

이번 모델은 Basic CNN과 Improved CNN보다 성능이 향상되었지만, 여전히 단순화된 VGG-style 구조입니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- 더 깊은 CNN 구조
- Learning Rate Scheduler 적용
- 다양한 Data Augmentation 추가
- Weight Decay 적용
- Residual Connection 학습
- ResNet 구조 구현

## Next Step

다음 단계에서는 ResNet의 핵심 구조인 BasicBlock과 Residual Connection을 학습하고 구현할 예정입니다.

이를 통해 단순히 깊게 쌓는 CNN 구조의 한계와, skip connection이 왜 필요한지 이해하는 것을 목표로 합니다.

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/intro.html
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
