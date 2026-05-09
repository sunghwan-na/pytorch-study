# FashionMNIST MLP Classifier

PyTorch를 사용하여 FashionMNIST 의류 이미지 데이터를 분류하는 기본 신경망 프로젝트입니다.

이 프로젝트의 목적은 높은 성능을 내는 것보다, PyTorch의 전체 학습 흐름을 직접 경험하는 것입니다.

## Project Goal

FashionMNIST 이미지를 입력받아 10개의 의류 클래스 중 하나로 분류하는 모델을 학습합니다.

## Dataset

사용한 데이터셋은 FashionMNIST입니다.

- 학습 데이터: 60,000개
- 테스트 데이터: 10,000개
- 이미지 크기: 28 × 28
- 이미지 형태: 흑백 이미지
- 클래스 수: 10개

클래스는 다음과 같습니다.

```text
0: T-Shirt
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle Boot
```

## Model Structure

이번 프로젝트에서는 CNN이 아닌 기본 MLP 모델을 사용했습니다.

모델 구조는 다음과 같습니다.

```text
Input Image: 1 × 28 × 28
→ Flatten: 784
→ Linear: 784 → 512
→ ReLU
→ Linear: 512 → 512
→ ReLU
→ Linear: 512 → 10
```

마지막 출력값 10개는 FashionMNIST의 10개 클래스를 의미합니다.

## Training Settings

| Setting | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |
| Batch Size | 64 |
| Epochs | 10 |
| Learning Rate | 1e-2 |
| Device | CPU |

## Experiment Results

| Model | Epochs | Learning Rate | Optimizer | Accuracy | Avg Loss |
|---|---:|---:|---|---:|---:|
| MLP | 5 | 1e-3 | SGD | 64.4% | 1.102602 |
| MLP | 10 | 1e-2 | SGD | 84.0% | 0.446630 |

## Result Analysis

처음에는 `epochs=5`, `learning rate=1e-3`으로 학습했을 때 정확도가 64.4%였습니다.

이후 `epochs=10`, `learning rate=1e-2`로 변경하자 정확도가 약 84.0%까지 상승했습니다.

이를 통해 학습 횟수와 learning rate가 모델 성능에 큰 영향을 준다는 것을 확인했습니다.

## Prediction Visualization

학습된 모델을 사용하여 테스트 이미지 9개를 랜덤으로 선택하고, 모델의 예측값과 실제 정답을 비교했습니다.

출력 예시는 다음과 같은 형태입니다.

```text
Pred: Bag
True: Bag

Pred: T-Shirt
True: T-Shirt

Pred: Shirt
True: Dress
```

`Pred`는 모델의 예측 결과이고, `True`는 실제 정답입니다.

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- FashionMNIST 데이터셋 불러오기
- DataLoader를 사용한 mini-batch 구성
- PyTorch `nn.Module`을 이용한 모델 정의
- `Flatten`, `Linear`, `ReLU`의 기본 사용법
- `CrossEntropyLoss`를 이용한 분류 손실 계산
- `SGD` optimizer를 이용한 가중치 업데이트
- train loop와 test loop의 기본 구조
- 모델 저장하기
- 예측 결과 시각화하기

## Key Training Flow

```text
데이터 불러오기
→ DataLoader 생성
→ 모델 정의
→ loss 함수 설정
→ optimizer 설정
→ 학습
→ 평가
→ 모델 저장
→ 예측 결과 시각화
```

## Limitations

이번 모델은 이미지를 CNN처럼 공간 구조로 분석하지 않고, 28×28 이미지를 784개의 숫자로 펼쳐서 학습합니다.

따라서 이미지의 공간적 특징을 충분히 활용하지 못한다는 한계가 있습니다.

## Next Step

다음 단계에서는 CNN 모델을 사용하여 FashionMNIST 분류 성능을 개선할 예정입니다.

CNN 버전에서는 다음 개념을 추가로 학습합니다.

- Conv2d
- ReLU
- MaxPool2d
- Feature Map
- CNN 기반 이미지 분류

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/intro.html
