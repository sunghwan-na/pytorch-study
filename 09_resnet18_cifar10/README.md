# ResNet-18 CIFAR-10 Classifier

PyTorch를 사용하여 CIFAR-10 이미지 데이터를 분류하는 ResNet-18 style CNN 프로젝트입니다.

이 프로젝트는 이전 `ResNet BasicBlock` 프로젝트에서 학습한 BasicBlock과 Shortcut Connection을 확장하여, ResNet-18에 가까운 구조를 직접 구현하는 것을 목표로 합니다.

## Project Goal

CIFAR-10 데이터셋을 사용하여 ResNet-18 style 모델을 구현하고, 이전 CNN 모델들과 성능을 비교합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- ResNet-18 style 구조 이해
- BasicBlock을 여러 개 쌓는 방식 학습
- `_make_layer()` 함수 이해
- Shortcut Connection을 이용한 깊은 CNN 구현
- 이전 모델들과 성능 비교

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

## BasicBlock

ResNet의 핵심은 입력 `x`를 Conv layer의 결과에 다시 더하는 Shortcut Connection입니다.

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

이를 통해 입력 정보가 뒤쪽 layer까지 잘 전달될 수 있고, 깊은 모델도 더 안정적으로 학습할 수 있습니다.

## Shortcut Connection

입력과 출력 shape가 같으면 shortcut은 입력 `x`를 그대로 사용합니다.

```text
shortcut(x) = x
```

하지만 채널 수나 이미지 크기가 달라지면 `out`과 `x`를 바로 더할 수 없습니다.

이 경우 shortcut 경로에 `1x1 Conv`를 사용하여 shape를 맞춥니다.

```python
nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
```

예를 들어:

```text
입력: [32, 32, 32]
출력: [64, 16, 16]
```

처럼 shape가 달라지는 경우, shortcut도 같은 shape로 변환한 뒤 더합니다.

## Model Structure

이번 프로젝트에서는 ResNet-18 style 구조를 사용했습니다.

```text
Input Image: 3 × 32 × 32

Initial Layer
→ Conv2d: 3 → 32
→ BatchNorm2d
→ ReLU

Layer 1
→ BasicBlock × 2
→ 32 channels, stride=1

Layer 2
→ BasicBlock × 2
→ 64 channels, stride=2

Layer 3
→ BasicBlock × 2
→ 128 channels, stride=2

Layer 4
→ BasicBlock × 2
→ 256 channels, stride=2

Classifier
→ AdaptiveAvgPool2d
→ Flatten
→ Linear: 256 → 10
```

마지막 출력값 10개는 CIFAR-10의 10개 클래스에 대한 예측 점수를 의미합니다.

## `_make_layer()` Function

이번 프로젝트에서는 `_make_layer()` 함수를 사용하여 BasicBlock을 반복해서 쌓았습니다.

```python
def _make_layer(self, out_channels, num_blocks, stride):
    layers = []

    layers.append(BasicBlock(self.in_channels, out_channels, stride))
    self.in_channels = out_channels

    for _ in range(1, num_blocks):
        layers.append(BasicBlock(self.in_channels, out_channels, stride=1))

    return nn.Sequential(*layers)
```

첫 번째 BasicBlock에서는 채널 수나 이미지 크기가 바뀔 수 있습니다.

그 이후 BasicBlock들은 같은 채널 수를 유지하면서 feature map을 처리합니다.

## Shape Flow

```text
Input: [3, 32, 32]

initial_layer
Conv2d(3, 32)
→ [32, 32, 32]

layer1 = BasicBlock × 2
→ 채널 수 유지, 이미지 크기 유지
→ [32, 32, 32]

layer2 = BasicBlock × 2
→ 채널 수 32에서 64로 증가
→ 이미지 크기 32×32에서 16×16으로 감소
→ [64, 16, 16]

layer3 = BasicBlock × 2
→ 채널 수 64에서 128로 증가
→ 이미지 크기 16×16에서 8×8로 감소
→ [128, 8, 8]

layer4 = BasicBlock × 2
→ 채널 수 128에서 256으로 증가
→ 이미지 크기 8×8에서 4×4로 감소
→ [256, 4, 4]

AdaptiveAvgPool2d((1, 1))
→ [256, 1, 1]

Flatten
→ 256

Linear(256, 10)
→ CIFAR-10의 10개 클래스 점수 출력
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
| ResNet-18 CIFAR-10 | 10 | Adam | 5e-4 | 84.5% | 0.473119 |

## Result Analysis

ResNet-18 style 모델은 84.5%의 정확도와 0.473119의 Avg Loss를 기록했습니다.

이는 이전에 구현한 Basic CNN, Improved CNN, VGG-style CNN, Simple ResNet BasicBlock 모델보다 높은 성능입니다.

BasicBlock을 여러 개 쌓고 Shortcut Connection을 사용한 ResNet 구조가 CIFAR-10 이미지 분류 성능 향상에 효과적임을 확인했습니다.

특히 이전 Simple ResNet BasicBlock보다 더 많은 BasicBlock을 체계적으로 쌓으면서 모델의 표현력이 향상되었습니다.

## Prediction Visualization

학습된 모델을 사용하여 테스트 이미지 9개를 랜덤으로 선택하고, 모델의 예측값과 실제 정답을 비교했습니다.

예측 결과는 다음과 같은 형태로 시각화했습니다.

```text
Pred: deer
True: deer

Pred: horse
True: horse

Pred: dog
True: cat
```

`Pred`는 모델의 예측값이고, `True`는 실제 정답입니다.

CIFAR-10은 32×32 크기의 저해상도 이미지이기 때문에 `cat`과 `dog`, `deer`와 `horse`, `automobile`과 `truck`, `airplane`과 `ship`처럼 형태가 비슷한 클래스에서는 오분류가 발생할 수 있습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- ResNet-18 style 구조
- BasicBlock을 여러 개 쌓는 방법
- `_make_layer()` 함수의 역할
- Shortcut Connection을 이용한 깊은 CNN 구조
- 입력과 출력 shape가 다를 때 1x1 Conv를 사용하는 이유
- AdaptiveAvgPool2d 사용법
- 이전 CNN 모델들과 ResNet-18 style 모델의 성능 비교
- 깊은 CNN 구조에서 Residual Connection이 중요한 이유

## Key Concepts

### ResNet-18 Style

ResNet-18은 BasicBlock을 여러 개 쌓아 구성한 깊은 CNN 구조입니다.

이번 프로젝트에서는 CIFAR-10에 맞게 단순화한 ResNet-18 style 모델을 구현했습니다.

### BasicBlock

BasicBlock은 두 개의 Conv layer를 통과한 결과에 shortcut 결과를 더하는 ResNet의 기본 단위입니다.

```text
output = F(x) + x
```

### Shortcut Connection

Shortcut Connection은 입력 `x`를 Conv 결과에 다시 더해주는 우회 경로입니다.

입력과 출력 shape가 같으면 그대로 더하고, shape가 다르면 1x1 Conv를 사용해 shape를 맞춥니다.

### `_make_layer()`

`_make_layer()`는 BasicBlock을 여러 개 반복해서 쌓기 위한 함수입니다.

이를 사용하면 layer1, layer2, layer3, layer4처럼 깊은 모델 구조를 더 깔끔하게 만들 수 있습니다.

## Limitations

이번 모델은 ResNet-18 구조를 CIFAR-10에 맞게 단순화한 모델입니다.

실제 ResNet-18은 더 정교한 구조와 학습 설정을 사용하며, 더 좋은 성능을 위해서는 추가적인 실험이 필요합니다.

개선할 수 있는 부분은 다음과 같습니다.

- Learning Rate Scheduler 적용
- Weight Decay 적용
- 더 다양한 Data Augmentation 적용
- 더 긴 epoch 학습
- GPU 환경에서 학습
- 실제 ResNet-18 구조와 비교 실험

## Next Step

다음 단계에서는 더 실제 논문 구조에 가까운 모델을 구현하거나, 컴퓨터비전의 다른 주요 task로 확장할 예정입니다.

후보는 다음과 같습니다.

- ResNet-18 구조 개선
- Learning Rate Scheduler 실험
- Object Detection 기초 실습
- Image Segmentation 기초 실습
- 논문 구조 따라 구현하기
