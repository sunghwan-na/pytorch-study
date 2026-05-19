# PyTorch Study

PyTorch 공식 튜토리얼과 컴퓨터비전 실습 프로젝트를 기반으로 기초 개념과 모델 학습 흐름을 정리하는 저장소입니다.

단순히 코드를 실행하는 것에서 끝내지 않고, 각 프로젝트별 핵심 개념과 실험 결과를 정리하며 컴퓨터비전 이미지 분류의 기본 흐름을 학습하는 것을 목표로 합니다.

## 학습 목표

- PyTorch 기본 문법 이해
- Tensor, Dataset, DataLoader, Transform 개념 정리
- 딥러닝 모델 학습 흐름 이해
- 이미지 분류 프로젝트를 통한 컴퓨터비전 기초 학습
- CNN 구조와 성능 개선 방법 학습
- VGG-style CNN 구조 이해
- ResNet BasicBlock과 Shortcut Connection 이해
- ResNet-18 style 구조 직접 구현
- Pretrained Model과 Transfer Learning 흐름 이해
- Fine-tuning 방식 이해
- Custom Dataset과 ImageFolder 사용법 학습
- 실험 결과를 GitHub에 정리하는 습관 만들기

## 학습 순서

1. Tensor
2. Dataset & DataLoader
3. Transform
4. FashionMNIST MLP Classifier
5. CIFAR-10 CNN Classifier
6. CIFAR-10 Improved CNN Classifier
7. VGG-style CNN Classifier
8. ResNet BasicBlock Classifier
9. ResNet-18 CIFAR-10 Classifier
10. Transfer Learning ResNet-18 Classifier
11. Fine-tuning ResNet-18 Classifier
12. Custom Dataset Fine-tuning Classifier

## 폴더 구조

```text
pytorch-study/
├── README.md
├── 01_tensor/
│   ├── tensor_basic.md
│   └── tensor_practice.py
├── 02_dataset_dataloader/
│   └── dataset_dataloader.md
├── 03_transform/
│   └── transform.md
├── 04_fashion_mnist_mlp/
│   ├── fashion_mnist_mlp.py
│   └── README.md
├── 05_cifar10_cnn/
│   ├── cifar10_cnn.py
│   └── README.md
├── 06_cifar10_cnn_improved/
│   ├── cifar10_cnn_improved.py
│   └── README.md
├── 07_vgg_style_cnn/
│   ├── vgg_style_cnn.py
│   └── README.md
├── 08_resnet_basicblock/
│   ├── resnet_basicblock.py
│   └── README.md
├── 09_resnet18_cifar10/
│   ├── resnet18_cifar10.py
│   └── README.md
├── 10_transfer_learning_resnet18/
│   ├── transfer_learning_resnet18.py
│   └── README.md
├── 11_finetuning_resnet18/
│   ├── finetuning_resnet18.py
│   └── README.md
└── 12_custom_dataset_finetuning/
    ├── custom_dataset_finetuning.py
    └── README.md
```

## 프로젝트 목록

### 1. FashionMNIST MLP Classifier

FashionMNIST 데이터셋을 사용하여 MLP 모델을 구현한 프로젝트입니다.

PyTorch의 기본 학습 흐름인 Dataset, DataLoader, Model, Loss Function, Optimizer, Train, Test 과정을 학습했습니다.

- Dataset: FashionMNIST
- Model: MLP
- 학습 목표: PyTorch 기본 학습 흐름 이해

### 2. CIFAR-10 CNN Classifier

CIFAR-10 데이터셋을 사용하여 기본 CNN 모델을 구현한 프로젝트입니다.

Conv2d, ReLU, MaxPool2d, Flatten, Linear layer를 사용하여 이미지 분류 모델의 기본 구조를 학습했습니다.

- Dataset: CIFAR-10
- Model: Basic CNN
- Accuracy: 71.7%
- Avg Loss: 1.054057
- 학습 목표: CNN 기본 구조 이해

### 3. CIFAR-10 Improved CNN Classifier

기본 CNN 모델에 성능 개선 요소를 추가한 프로젝트입니다.

Data Augmentation, BatchNorm2d, Dropout을 적용하여 모델 성능 개선 방법을 학습했습니다.

- Dataset: CIFAR-10
- Model: Improved CNN
- Accuracy: 75.3%
- Avg Loss: 0.703456
- 학습 목표: CNN 성능 개선 기법 이해

### 4. VGG-style CNN Classifier

VGG-style 구조를 참고하여 Conv layer를 여러 개 쌓은 CNN 모델을 구현한 프로젝트입니다.

Basic CNN보다 깊은 구조를 사용하여 feature extraction 능력을 향상시키는 방법을 학습했습니다.

- Dataset: CIFAR-10
- Model: VGG-style CNN
- Accuracy: 81.4%
- Avg Loss: 0.549206
- 학습 목표: 깊은 CNN 구조 이해

### 5. ResNet BasicBlock Classifier

ResNet의 핵심 구조인 BasicBlock과 Shortcut Connection을 직접 구현한 프로젝트입니다.

Conv 결과에 입력 x를 다시 더하는 Residual Connection 구조를 학습했습니다.

- Dataset: CIFAR-10
- Model: Simple ResNet BasicBlock
- Accuracy: 79.8%
- Avg Loss: 0.594762
- 학습 목표: BasicBlock과 Shortcut Connection 이해

### 6. ResNet-18 CIFAR-10 Classifier

BasicBlock을 여러 개 쌓아 ResNet-18 style 모델을 직접 구현한 프로젝트입니다.

`_make_layer()` 함수를 사용하여 BasicBlock을 반복해서 쌓는 구조를 학습했습니다.

- Dataset: CIFAR-10
- Model: ResNet-18 style CNN
- Accuracy: 84.5%
- Avg Loss: 0.473119
- 학습 목표: ResNet-18 style 구조 직접 구현

### 7. Transfer Learning ResNet-18 Classifier

torchvision에서 제공하는 pretrained ResNet-18 모델을 사용하여 CIFAR-10 데이터셋에 전이학습을 적용한 프로젝트입니다.

ImageNet으로 미리 학습된 ResNet-18의 기존 파라미터는 고정하고, 마지막 fc layer만 CIFAR-10의 10개 클래스에 맞게 수정하여 학습했습니다.

- Dataset: CIFAR-10
- Model: Pretrained ResNet-18
- Method: Feature Extractor
- Trainable Layers: fc layer
- Epochs: 3
- Accuracy: 79.6%
- Avg Loss: 0.615041
- 학습 목표: pretrained model 활용과 Feature Extractor 방식 이해

### 8. Fine-tuning ResNet-18 Classifier

pretrained ResNet-18 모델의 일부 layer를 함께 학습하는 Fine-tuning 프로젝트입니다.

Feature Extractor 방식과 다르게 `layer4`와 `fc layer`를 함께 학습하여 CIFAR-10 데이터셋에 더 적합하게 모델을 조정했습니다.

- Dataset: CIFAR-10
- Model: Pretrained ResNet-18
- Method: Fine-tuning
- Trainable Layers: layer4 + fc
- Epochs: 1
- Accuracy: 90.8%
- Avg Loss: 0.266824
- 학습 목표: Fine-tuning 방식 이해

### 9. Custom Dataset Fine-tuning Classifier

PyTorch의 `ImageFolder`를 사용하여 폴더 구조로 된 Custom Dataset을 불러오고, pretrained ResNet-18 모델을 fine-tuning한 프로젝트입니다.

CIFAR-10처럼 제공되는 데이터셋이 아니라, `dataset/train`, `dataset/val` 구조의 이미지 폴더를 직접 불러오는 방법을 학습했습니다.

- Dataset: Hymenoptera Dataset
- Classes: ants / bees
- Train Images: 244
- Validation Images: 153
- Model: Pretrained ResNet-18
- Method: Fine-tuning
- Trainable Layers: layer4 + fc
- Epochs: 1
- Accuracy: 88.9%
- Avg Loss: 0.264598
- 학습 목표: Custom Dataset과 ImageFolder 사용법 이해

## 프로젝트 성능 비교

| Project | Dataset | Model | Method | Accuracy | Avg Loss |
|---|---|---|---|---:|---:|
| FashionMNIST MLP | FashionMNIST | MLP | Train from scratch | 약 84% | 0.447070 |
| CIFAR-10 CNN | CIFAR-10 | Basic CNN | Train from scratch | 71.7% | 1.054057 |
| CIFAR-10 Improved CNN | CIFAR-10 | Improved CNN | Train from scratch | 75.3% | 0.703456 |
| VGG-style CNN | CIFAR-10 | VGG-style CNN | Train from scratch | 81.4% | 0.549206 |
| ResNet BasicBlock | CIFAR-10 | Simple ResNet | Train from scratch | 79.8% | 0.594762 |
| ResNet-18 CIFAR-10 | CIFAR-10 | ResNet-18 style | Train from scratch | 84.5% | 0.473119 |
| Transfer Learning ResNet-18 | CIFAR-10 | Pretrained ResNet-18 | Feature Extractor | 79.6% | 0.615041 |
| Fine-tuning ResNet-18 | CIFAR-10 | Pretrained ResNet-18 | Fine-tuning | 90.8% | 0.266824 |
| Custom Dataset Fine-tuning | Hymenoptera | Pretrained ResNet-18 | Fine-tuning | 88.9% | 0.264598 |

## 핵심 학습 내용

### PyTorch 기본 흐름

- Tensor
- Dataset
- DataLoader
- Transform
- Model 정의
- Loss Function
- Optimizer
- Train 함수
- Test / Validation 함수
- Model 저장

### CNN 기초

- Conv2d
- Feature Map
- ReLU
- MaxPool2d
- Flatten
- Linear layer
- CrossEntropyLoss
- Adam Optimizer

### CNN 성능 개선

- Data Augmentation
- BatchNorm2d
- Dropout
- deeper CNN structure
- VGG-style CNN

### ResNet 구조

- BasicBlock
- Shortcut Connection
- Residual Connection
- 1x1 Conv
- AdaptiveAvgPool2d
- `_make_layer()` 함수
- ResNet-18 style 구조

### Transfer Learning

- pretrained model
- ResNet18_Weights.DEFAULT
- ImageNet 기준 Transform
- fc layer 수정
- requires_grad=False
- Feature Extractor 방식
- Fine-tuning 방식

### Custom Dataset

- ImageFolder
- dataset/train, dataset/val 구조
- class_names
- class_to_idx
- num_classes = len(class_names)
- Custom Dataset에 맞는 fc layer 수정

## 학습 흐름 요약

```text
PyTorch 기초 개념 정리
→ FashionMNIST MLP로 전체 학습 흐름 경험
→ CIFAR-10 CNN으로 컬러 이미지 분류 학습
→ Improved CNN으로 성능 개선 기법 적용
→ VGG-style CNN으로 깊은 CNN 구조 구현
→ ResNet BasicBlock으로 shortcut connection 이해
→ ResNet-18 style 모델 직접 구현
→ pretrained ResNet-18을 활용한 Transfer Learning 실습
→ layer4 + fc를 학습하는 Fine-tuning 실습
→ ImageFolder를 사용한 Custom Dataset Fine-tuning 실습
```

## 정리

이 저장소는 PyTorch와 컴퓨터비전 이미지 분류의 기본 흐름을 단계적으로 학습한 기록입니다.

초기에는 FashionMNIST와 CIFAR-10을 사용하여 기본 학습 흐름과 CNN 구조를 익혔고, 이후 VGG-style CNN과 ResNet 구조를 직접 구현하며 깊은 CNN 모델의 구조를 학습했습니다.

그다음 pretrained ResNet-18을 활용하여 Transfer Learning과 Fine-tuning을 실습했고, 마지막으로 ImageFolder를 사용하여 Custom Dataset을 직접 불러와 fine-tuning하는 프로젝트를 진행했습니다.

이를 통해 단순한 이미지 분류 모델 구현에서 시작해, 실제 프로젝트에 가까운 Custom Dataset 학습 흐름까지 경험했습니다.

## 다음 학습 계획

- Object Detection 기초 실습
- Bounding Box 개념 학습
- pretrained detection model 사용
- Faster R-CNN 또는 YOLO 기초 실습
- Image Segmentation 기초 학습
- 논문 구조 따라 구현하기
