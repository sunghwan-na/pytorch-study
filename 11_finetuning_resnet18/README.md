# Fine-tuning ResNet-18 Classifier

PyTorch의 torchvision에서 제공하는 pretrained ResNet-18 모델을 사용하여 CIFAR-10 이미지 데이터를 분류하는 Fine-tuning 프로젝트입니다.

이 프로젝트는 이전 `Transfer Learning ResNet-18` 프로젝트에서 진행한 Feature Extractor 방식과 비교하기 위해, pretrained ResNet-18의 일부 layer를 함께 학습하는 것을 목표로 합니다.

## Project Goal

pretrained ResNet-18 모델을 불러온 뒤, 마지막 `fc layer`뿐만 아니라 `layer4`도 함께 학습하여 Fine-tuning 흐름을 이해합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- pretrained ResNet-18 모델 불러오기
- 기존 pretrained 파라미터 고정하기
- `layer4`와 `fc layer`만 학습 가능하게 설정하기
- Feature Extractor 방식과 Fine-tuning 방식 비교하기
- CIFAR-10 데이터셋에서 성능 변화 확인하기

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

pretrained ResNet-18은 ImageNet 기준으로 학습된 모델이기 때문에 CIFAR-10 이미지를 224×224 크기로 변환하고, ImageNet 기준 평균과 표준편차로 정규화했습니다.

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

학습 데이터에는 Data Augmentation을 적용하고, 테스트 데이터에는 평가 기준을 일정하게 유지하기 위해 랜덤 변형을 적용하지 않았습니다.

## Pretrained ResNet-18

torchvision에서 제공하는 pretrained ResNet-18 모델을 사용했습니다.

```python
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

이 모델은 ImageNet 데이터셋으로 미리 학습된 모델입니다.

## Fine-tuning 방식

이번 프로젝트에서는 pretrained ResNet-18의 전체 파라미터를 먼저 고정한 뒤, `layer4`와 `fc layer`만 학습하도록 설정했습니다.

```python
for param in model.parameters():
    param.requires_grad = False
```

이 코드는 기존 pretrained 모델의 파라미터를 업데이트하지 않도록 고정합니다.

그다음 `layer4`는 다시 학습 가능하게 설정했습니다.

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

마지막으로 CIFAR-10은 클래스가 10개이므로 `fc layer`를 수정했습니다.

```python
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
```

즉, 기존 ImageNet 1000개 클래스 출력 구조를 CIFAR-10의 10개 클래스 출력 구조로 바꾼 것입니다.

## Optimizer

이번 Fine-tuning 실험에서는 `layer4`와 `fc layer`만 학습하기 위해 optimizer에 두 layer의 파라미터만 전달했습니다.

```python
optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4
)
```

이전 Feature Extractor 방식에서는 `fc layer`만 학습했지만, 이번 Fine-tuning 방식에서는 `layer4`도 함께 학습합니다.

## Training Settings

| Setting | Value |
|---|---|
| Model | Pretrained ResNet-18 |
| Method | Fine-tuning |
| Trainable Layers | layer4 + fc |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 1 |
| Device | Google Colab GPU |
| Input Size | 224 × 224 |

## Experiment Results

| Model | Method | Epochs | Accuracy | Avg Loss |
|---|---|---:|---:|---:|
| Fine-tuning ResNet-18 | layer4 + fc | 1 | 90.8% | 0.266824 |

## Result Analysis

Fine-tuning ResNet-18 모델은 1 epoch 학습만으로 90.8%의 정확도와 0.266824의 Avg Loss를 기록했습니다.

이전 Transfer Learning ResNet-18 프로젝트에서는 pretrained ResNet-18의 기존 파라미터를 고정하고 마지막 `fc layer`만 학습했습니다.

반면 이번 Fine-tuning 프로젝트에서는 `layer4`와 `fc layer`를 함께 학습했습니다.

그 결과 CIFAR-10 데이터셋에 더 적합한 feature를 학습할 수 있었고, 정확도가 크게 향상되었습니다.

## Comparison with Previous Transfer Learning Project

| Model | Method | Trainable Layers | Epochs | Accuracy | Avg Loss |
|---|---|---|---:|---:|---:|
| Transfer Learning ResNet-18 | Feature Extractor | fc only | 3 | 79.6% | 0.615041 |
| Fine-tuning ResNet-18 | Fine-tuning | layer4 + fc | 1 | 90.8% | 0.266824 |

Feature Extractor 방식은 마지막 `fc layer`만 학습했기 때문에 빠르고 간단하지만, 성능 향상에는 한계가 있었습니다.

Fine-tuning 방식은 pretrained 모델의 일부 layer를 함께 학습하여 CIFAR-10 데이터셋에 더 잘 맞게 조정할 수 있었습니다.

## Comparison with Previous Projects

| Model | Dataset | Method | Accuracy | Avg Loss |
|---|---|---|---:|---:|
| Basic CNN | CIFAR-10 | Train from scratch | 71.7% | 1.054057 |
| Improved CNN | CIFAR-10 | Train from scratch | 75.3% | 0.703456 |
| VGG-style CNN | CIFAR-10 | Train from scratch | 81.4% | 0.549206 |
| Simple ResNet BasicBlock | CIFAR-10 | Train from scratch | 79.8% | 0.594762 |
| ResNet-18 CIFAR-10 | CIFAR-10 | Train from scratch | 84.5% | 0.473119 |
| Transfer Learning ResNet-18 | CIFAR-10 | Feature Extractor | 79.6% | 0.615041 |
| Fine-tuning ResNet-18 | CIFAR-10 | Fine-tuning | 90.8% | 0.266824 |

Fine-tuning ResNet-18은 지금까지 진행한 CIFAR-10 이미지 분류 프로젝트 중 가장 높은 정확도를 기록했습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- pretrained ResNet-18 모델을 불러오는 방법
- ImageNet 기준 Transform을 적용하는 이유
- CIFAR-10 이미지를 224×224로 Resize하는 이유
- `requires_grad = False`로 파라미터를 고정하는 방법
- `layer4`만 다시 학습 가능하게 설정하는 방법
- `fc layer`를 CIFAR-10 클래스 수에 맞게 수정하는 방법
- Feature Extractor와 Fine-tuning의 차이
- Fine-tuning이 성능 향상에 도움이 될 수 있다는 점

## Key Concepts

### Fine-tuning

Fine-tuning은 pretrained 모델의 일부 layer를 새로운 데이터셋에 맞게 다시 학습하는 방법입니다.

이번 프로젝트에서는 ResNet-18의 `layer4`와 `fc layer`를 학습했습니다.

### Feature Extractor와 Fine-tuning 차이

Feature Extractor 방식은 pretrained 모델의 대부분 파라미터를 고정하고 마지막 classifier만 학습합니다.

Fine-tuning 방식은 pretrained 모델의 일부 layer도 함께 학습하여 새로운 데이터셋에 더 잘 맞게 조정합니다.

### layer4

`layer4`는 ResNet-18의 마지막 feature extraction layer입니다.

이미지의 고수준 특징을 추출하는 부분이기 때문에, 이 layer를 함께 학습하면 CIFAR-10 데이터셋에 더 적합한 특징을 학습할 수 있습니다.

### fc layer

`fc layer`는 ResNet-18의 마지막 분류층입니다.

ResNet-18은 원래 ImageNet 1000개 클래스를 분류하도록 학습되었지만, CIFAR-10은 10개 클래스이므로 `Linear(512, 10)`으로 수정했습니다.

## Limitations

이번 실험은 Fine-tuning의 기본 흐름을 이해하기 위한 실험입니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- epoch 수 증가
- Learning Rate Scheduler 적용
- Weight Decay 적용
- 더 다양한 Data Augmentation 적용
- layer3까지 함께 Fine-tuning하는 실험
- 다른 pretrained 모델과 비교

## Next Step

다음 단계에서는 Fine-tuning 범위를 조절하며 성능 변화를 비교할 수 있습니다.

예를 들어 다음과 같은 실험을 할 수 있습니다.

- fc만 학습
- layer4 + fc 학습
- layer3 + layer4 + fc 학습

또는 이미지 분류를 넘어 Object Detection이나 Image Segmentation 기초 프로젝트로 확장할 수 있습니다.
