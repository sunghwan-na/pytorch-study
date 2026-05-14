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

## Dataset

사용한 데이터셋은 CIFAR-10입니다.

- 학습 데이터: 50,000개
- 테스트 데이터: 10,000개
- 원본 이미지 크기: 32 × 32
- 변환 후 이미지 크기: 224 × 224
- 이미지 형태: RGB 컬러 이미지
- 클래스 수: 10개

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
| Epochs | 1 |
| Device | CPU |
| Input Size | 224 × 224 |

## Experiment Results

| Model | Method | Epochs | Accuracy | Avg Loss |
|---|---|---:|---:|---:|
| Transfer Learning ResNet-18 | Feature Extractor | 1 | 78.1% | 0.636990 |

## Result Analysis

Feature Extractor 방식으로 pretrained ResNet-18의 기존 파라미터는 고정하고, 마지막 `fc layer`만 CIFAR-10에 맞게 학습했습니다.

1 epoch만 학습했음에도 정확도 78.1%, Avg Loss 0.636990을 기록했습니다.

이는 모델을 처음부터 랜덤하게 학습한 것이 아니라, ImageNet으로 미리 학습된 ResNet-18의 feature extraction 능력을 활용했기 때문에 가능한 결과입니다.

다만 이번 실험은 CPU 환경에서 진행되었고, 학습 시간이 오래 걸리기 때문에 epoch를 1로 제한했습니다. 따라서 최종 성능을 최대한 높이는 실험보다는 전이학습의 기본 흐름을 이해하는 데 목적이 있습니다.

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

## Key Concepts

### Transfer Learning

Transfer Learning은 이미 큰 데이터셋으로 학습된 모델을 가져와 새로운 데이터셋에 맞게 활용하는 방법입니다.

이번 프로젝트에서는 ImageNet으로 학습된 ResNet-18을 CIFAR-10 분류에 사용했습니다.

### Feature Extractor

Feature Extractor 방식은 pretrained 모델의 대부분 파라미터를 고정하고, 마지막 classifier만 새 데이터셋에 맞게 학습하는 방식입니다.

### Fine-tuning과의 차이

Feature Extractor 방식은 마지막 분류층만 학습합니다.

반면 Fine-tuning은 pretrained 모델의 일부 또는 전체 layer를 다시 학습합니다.

이번 프로젝트에서는 CPU 환경을 고려하여 Feature Extractor 방식만 사용했습니다.

## Limitations

이번 실험은 CPU 환경에서 진행되었기 때문에 학습 시간이 오래 걸렸습니다.

또한 epoch를 1로 제한했기 때문에 충분히 학습된 결과는 아닙니다.

더 높은 성능을 위해서는 다음과 같은 개선이 필요합니다.

- epoch 수 증가
- GPU 환경 사용
- Fine-tuning 적용
- Learning Rate Scheduler 적용
- 더 다양한 Data Augmentation 적용

## Next Step

다음 단계에서는 GPU 환경이 가능할 때 Fine-tuning 방식을 적용해볼 수 있습니다.

또는 전이학습을 다른 데이터셋에 적용하여, 실제 이미지 분류 프로젝트에 가까운 형태로 확장할 수 있습니다.
