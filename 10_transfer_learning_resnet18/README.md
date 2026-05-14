# Transfer Learning ResNet-18 Classifier

PyTorch의 torchvision에서 제공하는 pretrained ResNet-18 모델을 사용하여 CIFAR-10 이미지 데이터를 분류하는 전이학습 프로젝트입니다.

이 프로젝트는 직접 모델을 처음부터 구현하는 것이 아니라, ImageNet으로 미리 학습된 ResNet-18 모델을 불러와 마지막 분류층만 CIFAR-10에 맞게 수정하는 것을 목표로 합니다.

## Project Goal

pretrained ResNet-18 모델을 사용하여 전이학습의 기본 흐름을 이해합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- pretrained model 불러오기
- ResNet-18의 마지막 fc layer 수정하기
- 기존 pretrained 파라미터 고정하기
- 마지막 classifier만 학습하는 Feature Extractor 방식 이해하기
- CIFAR-10 데이터셋에 전이학습 적용하기
- epoch 변화에 따른 성능 확인하기

## Dataset

사용한 데이터셋은 CIFAR-10입니다.

- 학습 데이터: 50,000개
- 테스트 데이터: 10,000개
- 원본 이미지 크기: 32 × 32
- 변환 후 이미지 크기: 224 × 224
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

pretrained ResNet-18은 ImageNet 기준으로 학습된 모델이기 때문에, CIFAR-10 이미지를 224×224 크기로 변환하고 ImageNet 기준 평균과 표준편차로 정규화했습니다.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
```

학습 데이터에는 `RandomHorizontalFlip`과 `RandomCrop`을 적용하여 Data Augmentation을 사용했습니다.

테스트 데이터에는 평가 기준을 일정하게 유지하기 위해 랜덤 변형을 적용하지 않았습니다.

## Pretrained ResNet-18

torchvision에서 제공하는 pretrained ResNet-18 모델을 사용했습니다.

```python
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

이 모델은 ImageNet 데이터셋으로 미리 학습된 모델입니다.

## Feature Extractor 방식

이번 프로젝트에서는 기존 pretrained ResNet-18의 파라미터를 고정하고, 마지막 `fc layer`만 학습했습니다.

```python
for param in model.parameters():
    param.requires_grad = False
```

이 코드는 기존 ResNet-18의 Conv layer, BatchNorm layer 등의 파라미터가 학습 중 업데이트되지 않도록 고정합니다.

이후 CIFAR-10은 클래스가 10개이므로 마지막 `fc layer`를 수정했습니다.

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

즉, 기존 ImageNet 1000개 클래스 출력 구조를 CIFAR-10의 10개 클래스 출력 구조로 바꾼 것입니다.

## Optimizer

마지막 `fc layer`만 학습하기 위해 optimizer에는 `model.fc.parameters()`만 전달했습니다.

```python
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

즉, 이번 실험에서는 전체 모델을 학습한 것이 아니라 마지막 classifier만 학습했습니다.

## Training Settings

| Setting | Value |
|---|---|
| Model | Pretrained ResNet-18 |
| Method | Feature Extractor |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Batch Size | 32 |
| Epochs | 3 |
| Device | Google Colab |
| Input Size | 224 × 224 |

## Experiment Results

| Epoch | Accuracy | Avg Loss |
|---:|---:|---:|
| 1 | 78.1% | 0.643030 |
| 2 | 79.4% | 0.600895 |
| 3 | 79.6% | 0.615041 |

## Best Result

| Model | Method | Epochs | Best Accuracy | Avg Loss |
|---|---|---:|---:|---:|
| Transfer Learning ResNet-18 | Feature Extractor | 3 | 79.6% | 0.615041 |

## Result Analysis

Feature Extractor 방식으로 pretrained ResNet-18의 기존 파라미터는 고정하고, 마지막 `fc layer`만 CIFAR-10에 맞게 학습했습니다.

1 epoch에서는 정확도 78.1%, Avg Loss 0.643030을 기록했습니다.

이후 3 epoch까지 학습한 결과 정확도는 79.6%까지 상승했습니다.

```text
Accuracy: 78.1% → 79.4% → 79.6%
```

Avg Loss는 Epoch 2에서 가장 낮은 0.600895를 기록했고, Epoch 3에서는 정확도가 가장 높았습니다.

```text
Avg Loss: 0.643030 → 0.600895 → 0.615041
```

이번 실험을 통해 pretrained ResNet-18의 feature extraction 능력을 활용하면 적은 epoch만으로도 비교적 높은 정확도를 얻을 수 있음을 확인했습니다.

다만 이번 방식은 기존 ResNet-18의 대부분 파라미터를 고정하고 마지막 classifier만 학습했기 때문에, 성능 향상에는 한계가 있습니다.

더 높은 성능을 위해서는 일부 layer를 함께 학습하는 Fine-tuning 방식이 필요할 수 있습니다.

## Comparison with Previous Projects

| Model | Dataset | Method | Accuracy | Avg Loss |
|---|---|---|---:|---:|
| Basic CNN | CIFAR-10 | Train from scratch | 71.7% | 1.054057 |
| Improved CNN | CIFAR-10 | Train from scratch | 75.3% | 0.703456 |
| VGG-style CNN | CIFAR-10 | Train from scratch | 81.4% | 0.549206 |
| Simple ResNet BasicBlock | CIFAR-10 | Train from scratch | 79.8% | 0.594762 |
| ResNet-18 CIFAR-10 | CIFAR-10 | Train from scratch | 84.5% | 0.473119 |
| Transfer Learning ResNet-18 | CIFAR-10 | Feature Extractor | 79.6% | 0.615041 |

직접 구현한 ResNet-18 CIFAR-10 모델이 더 높은 정확도를 기록했지만, Transfer Learning ResNet-18은 마지막 classifier만 학습했음에도 3 epoch에서 79.6%의 정확도를 기록했습니다.

이를 통해 pretrained model을 활용하면 적은 학습으로도 의미 있는 성능을 얻을 수 있음을 확인했습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- pretrained model을 불러오는 방법
- `ResNet18_Weights.DEFAULT` 사용법
- ImageNet 기준 Transform 적용
- CIFAR-10 이미지를 224×224로 Resize하는 이유
- 마지막 `fc layer`를 CIFAR-10 클래스 수에 맞게 수정하는 방법
- `requires_grad = False`로 기존 파라미터를 고정하는 방법
- `model.fc.parameters()`만 optimizer에 전달하여 classifier만 학습하는 방법
- Feature Extractor 방식의 전이학습 흐름
- epoch 변화에 따른 accuracy와 loss 변화 확인

## Key Concepts

### Transfer Learning

Transfer Learning은 이미 큰 데이터셋으로 학습된 모델을 가져와 새로운 데이터셋에 맞게 활용하는 방법입니다.

이번 프로젝트에서는 ImageNet으로 학습된 ResNet-18을 CIFAR-10 분류에 사용했습니다.

### Feature Extractor

Feature Extractor 방식은 pretrained 모델의 대부분 파라미터를 고정하고, 마지막 classifier만 새 데이터셋에 맞게 학습하는 방식입니다.

이번 프로젝트에서는 ResNet-18의 기존 feature extraction 능력은 그대로 사용하고, 마지막 `fc layer`만 CIFAR-10에 맞게 학습했습니다.

### Fine-tuning과의 차이

Feature Extractor 방식은 마지막 분류층만 학습합니다.

반면 Fine-tuning은 pretrained 모델의 일부 또는 전체 layer를 다시 학습합니다.

이번 프로젝트에서는 Feature Extractor 방식을 사용했고, 다음 단계에서는 Fine-tuning을 적용할 수 있습니다.

## Limitations

이번 실험은 Feature Extractor 방식으로 진행되었기 때문에 기존 pretrained ResNet-18의 대부분 파라미터는 업데이트되지 않았습니다.

따라서 CIFAR-10 데이터셋에 완전히 맞게 모델 전체가 조정된 것은 아닙니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- epoch 수 증가
- Fine-tuning 적용
- layer4와 fc layer 함께 학습
- Learning Rate Scheduler 적용
- 더 다양한 Data Augmentation 적용
- GPU 환경에서 더 긴 학습 진행

## Next Step

다음 단계에서는 pretrained ResNet-18의 일부 layer를 함께 학습하는 Fine-tuning 방식을 적용해볼 수 있습니다.

특히 `layer4`와 `fc layer`만 학습하도록 설정하면, Feature Extractor 방식보다 CIFAR-10에 더 잘 맞는 모델을 만들 수 있습니다.

또는 전이학습을 다른 이미지 데이터셋에 적용하여 실제 이미지 분류 프로젝트에 가까운 형태로 확장할 수 있습니다.
