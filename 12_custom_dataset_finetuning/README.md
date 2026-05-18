# Custom Dataset Fine-tuning Classifier

PyTorch의 `ImageFolder`를 사용하여 폴더 구조로 된 Custom Dataset을 불러오고, pretrained ResNet-18 모델을 fine-tuning하여 이미지 분류를 수행한 프로젝트입니다.

이번 프로젝트는 CIFAR-10처럼 PyTorch에서 제공하는 데이터셋을 사용하는 것이 아니라, 직접 준비한 이미지 폴더 구조를 데이터셋으로 사용하는 방법을 학습하는 것을 목표로 합니다.

## Project Goal

Custom Dataset을 직접 불러오고 pretrained ResNet-18을 fine-tuning하는 전체 흐름을 이해합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- `ImageFolder`를 사용한 Custom Dataset 불러오기
- 폴더 이름을 class label로 사용하는 구조 이해
- `train` / `val` 데이터셋 분리
- `class_names`, `class_to_idx` 확인
- Custom Dataset 클래스 수에 맞게 `fc layer` 수정
- pretrained ResNet-18의 `layer4 + fc layer` fine-tuning

## Dataset

이번 프로젝트에서는 PyTorch 튜토리얼에서 사용하는 Hymenoptera Dataset을 사용했습니다.

- Classes: ants, bees
- Train images: 244
- Validation images: 153
- Input image size: 224 × 224
- Model: pretrained ResNet-18

## Dataset Structure

`ImageFolder`는 폴더 이름을 기준으로 class label을 자동으로 생성합니다.

사용한 폴더 구조는 다음과 같습니다.

```text
dataset/
├── train/
│   ├── ants/
│   └── bees/
└── val/
    ├── ants/
    └── bees/
```

예를 들어 `dataset/train/ants` 폴더 안의 이미지는 `ants` 클래스로 인식되고, `dataset/train/bees` 폴더 안의 이미지는 `bees` 클래스로 인식됩니다.

## Dataset Download

Colab에서 다음 코드를 사용하여 예제 데이터셋을 다운로드했습니다.

```python
!wget -q https://download.pytorch.org/tutorial/hymenoptera_data.zip
!unzip -q hymenoptera_data.zip
!mv hymenoptera_data dataset
```

폴더 구조 확인:

```python
!find dataset -maxdepth 2 -type d
```

정상적인 구조는 다음과 같습니다.

```text
dataset
dataset/train
dataset/train/ants
dataset/train/bees
dataset/val
dataset/val/ants
dataset/val/bees
```

## Transform

pretrained ResNet-18은 ImageNet 기준으로 학습된 모델이기 때문에, 입력 이미지를 224×224 크기로 변환하고 ImageNet 기준 평균과 표준편차로 정규화했습니다.

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])
```

학습 데이터에는 `RandomHorizontalFlip`과 `RandomRotation`을 적용하여 Data Augmentation을 사용했습니다.

검증 데이터에는 평가 기준을 일정하게 유지하기 위해 랜덤 변형을 적용하지 않았습니다.

## ImageFolder

Custom Dataset은 `datasets.ImageFolder`를 사용하여 불러왔습니다.

```python
train_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

val_data = datasets.ImageFolder(
    root=val_dir,
    transform=val_transform
)
```

`ImageFolder`는 폴더 이름을 class label로 사용합니다.

```python
class_names = train_data.classes
print(class_names)
print(train_data.class_to_idx)
```

출력 결과:

```text
['ants', 'bees']
{'ants': 0, 'bees': 1}
```

즉, `ants`는 label 0, `bees`는 label 1로 변환됩니다.

## Model

pretrained ResNet-18 모델을 사용했습니다.

```python
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
```

기존 pretrained 파라미터를 먼저 고정했습니다.

```python
for param in model.parameters():
    param.requires_grad = False
```

그 후 `layer4`만 다시 학습 가능하게 설정했습니다.

```python
for param in model.layer4.parameters():
    param.requires_grad = True
```

Custom Dataset은 클래스 수가 고정되어 있지 않기 때문에, `class_names`의 길이를 사용하여 출력 클래스 수를 정했습니다.

```python
num_features = model.fc.in_features
num_classes = len(class_names)
model.fc = nn.Linear(num_features, num_classes)
```

이번 데이터셋은 `ants`, `bees` 2개 클래스이므로 마지막 `fc layer`는 다음과 같은 구조가 됩니다.

```text
Linear(512, 2)
```

## Fine-tuning Setting

이번 프로젝트에서는 pretrained ResNet-18의 `layer4`와 `fc layer`만 학습했습니다.

```python
optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4
)
```

즉, 전체 모델을 처음부터 학습한 것이 아니라, pretrained 모델의 일부 layer만 Custom Dataset에 맞게 조정했습니다.

## Training Settings

| Setting | Value |
|---|---|
| Dataset | Hymenoptera Dataset |
| Classes | ants / bees |
| Train Images | 244 |
| Validation Images | 153 |
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

| Dataset | Classes | Train Images | Validation Images | Epochs | Accuracy | Avg Loss |
|---|---|---:|---:|---:|---:|---:|
| Hymenoptera | ants / bees | 244 | 153 | 1 | 88.9% | 0.264598 |

## Result Analysis

Custom Dataset을 `ImageFolder`로 불러오고, pretrained ResNet-18을 fine-tuning한 결과 1 epoch 학습만으로 88.9%의 validation accuracy를 기록했습니다.

이번 데이터셋은 학습 이미지가 244개로 비교적 적지만, ImageNet으로 미리 학습된 ResNet-18을 사용했기 때문에 적은 데이터에서도 의미 있는 성능을 얻을 수 있었습니다.

특히 `layer4`와 `fc layer`만 학습하여 기존 pretrained 모델의 feature extraction 능력을 활용하면서도, ants/bees 데이터셋에 맞게 일부 layer를 조정했습니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- Custom Dataset 폴더 구조
- `ImageFolder` 사용법
- 폴더 이름이 class label로 사용되는 방식
- `class_names`와 `class_to_idx` 확인 방법
- `len(train_data)`, `len(val_data)`로 데이터 개수 확인하기
- Custom Dataset 클래스 수에 맞게 `fc layer` 수정하기
- pretrained ResNet-18을 Custom Dataset에 fine-tuning하기
- `layer4 + fc layer`만 학습하는 방법
- validation 데이터로 모델 성능을 확인하는 방법

## Key Concepts

### Custom Dataset

Custom Dataset은 사용자가 직접 준비한 이미지 데이터를 말합니다.

이번 프로젝트에서는 `dataset/train`, `dataset/val` 폴더를 만들고, 그 안에 클래스별 폴더를 구성했습니다.

### ImageFolder

`ImageFolder`는 폴더 구조를 보고 이미지 데이터셋을 만들어주는 PyTorch 도구입니다.

클래스 이름은 폴더 이름에서 가져오고, 각 클래스는 자동으로 label 번호로 변환됩니다.

### class_names

```python
class_names = train_data.classes
```

현재 데이터셋의 클래스 이름 목록을 저장합니다.

예:

```text
['ants', 'bees']
```

### class_to_idx

```python
train_data.class_to_idx
```

클래스 이름이 어떤 label 번호로 변환되었는지 보여줍니다.

예:

```text
{'ants': 0, 'bees': 1}
```

### num_classes

```python
num_classes = len(class_names)
```

Custom Dataset은 클래스 수가 데이터셋마다 다르기 때문에, 직접 숫자를 고정하지 않고 `len(class_names)`로 클래스 수를 계산했습니다.

### Fine-tuning

Fine-tuning은 pretrained 모델의 일부 layer를 새로운 데이터셋에 맞게 다시 학습하는 방법입니다.

이번 프로젝트에서는 ResNet-18의 `layer4`와 `fc layer`를 학습했습니다.

## Limitations

이번 실험은 Custom Dataset과 ImageFolder 사용법을 익히기 위한 기초 실험입니다.

데이터셋 크기가 작고 epoch도 1로 제한했기 때문에, 더 안정적인 성능 비교를 위해서는 추가 실험이 필요합니다.

개선할 수 있는 부분은 다음과 같습니다.

- epoch 수 증가
- 더 다양한 Data Augmentation 적용
- layer3까지 함께 fine-tuning
- 더 큰 Custom Dataset 사용
- train/val/test 분리
- 예측 이미지 시각화 추가

## Next Step

다음 단계에서는 Custom Dataset에 대해 더 긴 epoch로 학습하거나, 다른 이미지 데이터셋을 직접 구성해볼 수 있습니다.

또한 이미지 분류를 넘어 Object Detection이나 Image Segmentation 기초 프로젝트로 확장할 수 있습니다.
