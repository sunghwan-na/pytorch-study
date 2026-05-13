# ResNet BasicBlock Classifier

PyTorch를 사용하여 CIFAR-10 이미지 데이터를 분류하는 ResNet BasicBlock 구현 프로젝트입니다.

이 프로젝트는 ResNet의 핵심 구조인 Residual Connection과 Shortcut Connection을 직접 구현하며 이해하는 것을 목표로 합니다.

## Project Goal

CIFAR-10 데이터셋을 사용하여 간단한 ResNet 구조를 구현하고, BasicBlock이 어떻게 동작하는지 학습합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- ResNet BasicBlock 구조 이해
- Residual Connection / Shortcut Connection 개념 이해
- 입력과 출력 shape가 다를 때 1x1 Conv로 shape를 맞추는 과정 이해
- Simple ResNet 모델 구현
- 이전 CNN 모델들과 성능 비교

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

학습 데이터에는 Data Augmentation을 적용하고, 테스트 데이터에는 평가 기준을 일정하게 유지하기 위해 랜덤 변형을 적용하지 않았습니다.

## BasicBlock Structure

ResNet BasicBlock은 Conv layer를 통과한 결과에 입력 `x`를 다시 더하는 구조입니다.

일반 CNN은 다음과 같은 흐름을 가집니다.

```text
x → Conv → Conv → output
```

ResNet BasicBlock은 다음과 같은 흐름을 가집니다.

```text
x → Conv → Conv → out
↘──────────────↗
     shortcut
```

핵심 코드는 다음과 같습니다.

```python
out = self.conv_layer(x)
shortcut = self.shortcut(x)
out = out + shortcut
out = self.relu(out)
```

즉, Conv 결과에 shortcut 경로를 통해 전달된 입력 정보를 다시 더합니다.

## Shortcut Connection

Shortcut은 입력 `x`를 Conv 계산 결과에 다시 더해주는 우회 경로입니다.

입력과 출력 shape가 같으면 shortcut은 입력 `x`를 그대로 사용합니다.

```text
shortcut(x) = x
```

하지만 채널 수나 이미지 크기가 달라지면 `out`과 `x`를 바로 더할 수 없습니다.

예를 들어 다음 두 텐서는 shape가 다르기 때문에 더할 수 없습니다.

```text
out:      [64, 16, 16]
shortcut: [32, 32, 32]
```

이 경우 shortcut 경로에 `1x1 Conv`를 사용하여 shape를 맞춥니다.

```python
nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
```

이를 통해 shortcut의 shape를 Conv 결과와 같게 만든 후 더할 수 있습니다.

## Model Structure

이번 프로젝트에서는 간단한 Simple ResNet 구조를 사용했습니다.

```text
Input Image: 3 × 32 × 32

Initial Layer
→ Conv2d: 3 → 32
→ BatchNorm2d
→ ReLU

Layer 1
→ BasicBlock: 32 → 32, stride=1

Layer 2
→ BasicBlock: 32 → 64, stride=2

Layer 3
→ BasicBlock: 64 → 128, stride=2

Classifier
→ AdaptiveAvgPool2d
→ Flatten
→ Linear: 128 → 10
```

마지막 출력값 10개는 CIFAR-10의 10개 클래스에 대한 예측 점수를 의미합니다.

## Shape Flow

```text
Input: [3, 32, 32]

initial_layer
Conv2d(3, 32)
→ [32, 32, 32]

layer1 = BasicBlock(32, 32, stride=1)
→ 채널 수 유지, 이미지 크기 유지
→ [32, 32, 32]

layer2 = BasicBlock(32, 64, stride=2)
→ 채널 수 32에서 64로 증가
→ 이미지 크기 32×32에서 16×16으로 감소
→ [64, 16, 16]

layer3 = BasicBlock(64, 128, stride=2)
→ 채널 수 64에서 128로 증가
→ 이미지 크기 16×16에서 8×8로 감소
→ [128, 8, 8]

AdaptiveAvgPool2d((1, 1))
→ [128, 1, 1]

Flatten
→ 128

Linear(128, 10)
→ CIFAR-10의 10개 클래스 점수 출력
```

## AdaptiveAvgPool2d

이번 모델에서는 `AdaptiveAvgPool2d((1, 1))`을 사용했습니다.

```python
nn.AdaptiveAvgPool2d((1, 1))
```

이 층은 feature map의 공간 크기를 지정한 크기로 줄여줍니다.

이번 코드에서는 다음과 같이 변환됩니다.

```text
[128, 8, 8] → [128, 1, 1]
```

그 후 Flatten을 적용하면 다음과 같습니다.

```text
[128, 1, 1] → 128
```

그래서 마지막 Linear layer는 다음과 같이 작성할 수 있습니다.

```python
nn.Linear(128, 10)
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
| Simple ResNet BasicBlock | 10 | Adam | 5e-4 | 79.8% | 0.594762 |

## Result Analysis

Simple ResNet BasicBlock 모델은 79.8%의 정확도와 0.594762의 Avg Loss를 기록했습니다.

이는 Basic CNN과 Improved CNN보다 높은 결과입니다.

이를 통해 shortcut connection을 사용하는 ResNet 구조가 모델 학습과 성능 향상에 도움이 될 수 있음을 확인했습니다.

다만 이번 모델은 간단한 Simple ResNet 구조이기 때문에 VGG-style CNN보다 정확도는 조금 낮게 나왔습니다.

이번 프로젝트의 핵심 목적은 최고 성능 달성보다 Residual Connection과 BasicBlock 구조를 직접 구현하고 이해하는 것입니다.

## Prediction Visualization

학습된 모델을 사용하여 테스트 이미지 9개를 랜덤으로 선택하고, 모델의 예측값과 실제 정답을 비교했습니다.

예측 결과는 다음과 같은 형태로 시각화했습니다.

```text
Pred: deer
True: deer

Pred: ship
True: ship

Pred: dog
True: cat
```

`Pred`는 모델의 예측값이고, `True`는 실제 정답입니다.

CIFAR-10은 32×32 크기의 저해상도 이미지이기 때문에 `cat`과 `dog`, `deer`와 `horse`, `ship`과 `airplane`처럼 형태가 비슷한 클래스는 혼동될 수 있습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- ResNet BasicBlock 구조
- Residual Connection 개념
- Shortcut Connection 개념
- Conv 결과와 입력 x를 더하는 구조
- 입력과 출력 shape가 다를 때 1x1 Conv를 사용하는 이유
- BasicBlock을 여러 개 쌓아 Simple ResNet 모델을 만드는 방법
- AdaptiveAvgPool2d 사용법
- ResNet 구조와 기존 CNN 구조의 차이
- CIFAR-10 이미지 분류 성능 비교

## Key Concepts

### Residual Connection

Residual Connection은 입력 `x`를 Conv layer의 출력에 다시 더하는 구조입니다.

```text
output = F(x) + x
```

이를 통해 입력 정보가 뒤쪽 layer까지 잘 전달될 수 있습니다.

### Shortcut Connection

Shortcut Connection은 입력 `x`를 Conv 계산 결과에 더하기 위해 우회해서 보내는 경로입니다.

입력과 출력 shape가 같으면 그대로 더하고, shape가 다르면 `1x1 Conv`를 사용해 shape를 맞춥니다.

### BasicBlock

BasicBlock은 ResNet의 기본 구성 단위입니다.

두 개의 Conv layer를 거친 결과에 shortcut 결과를 더한 뒤 ReLU를 적용합니다.

```python
out = self.conv_layer(x)
shortcut = self.shortcut(x)
out = out + shortcut
out = self.relu(out)
```

## Limitations

이번 모델은 ResNet의 핵심 구조인 BasicBlock을 단순화해서 구현한 모델입니다.

실제 ResNet은 더 많은 block을 깊게 쌓고, 더 체계적인 layer 구성을 사용합니다.

따라서 이번 프로젝트는 실제 ResNet 성능을 재현하기보다는 Residual Connection과 BasicBlock 구조를 이해하는 데 목적이 있습니다.

## Next Step

다음 단계에서는 실제 ResNet 구조를 더 깊게 구현하거나, ResNet 논문의 구조를 참고하여 더 완성도 있는 모델을 구현할 예정입니다.

추가로 학습할 개념은 다음과 같습니다.

- Residual Learning
- Deeper ResNet
- ResNet-18 구조
- Learning Rate Scheduler
- Weight Decay
- 논문 구조 구현 연습
