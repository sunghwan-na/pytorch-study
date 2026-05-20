# Object Detection Basic

PyTorch의 torchvision에서 제공하는 pretrained Faster R-CNN 모델을 사용하여 이미지 한 장에서 객체를 탐지하고, bounding box를 시각화하는 프로젝트입니다.

이번 프로젝트는 이미지 분류를 넘어, 이미지 안의 객체 위치와 클래스를 함께 예측하는 Object Detection의 기본 흐름을 이해하는 것을 목표로 합니다.

## Project Goal

pretrained Faster R-CNN 모델을 사용하여 객체 탐지의 기본 출력 구조를 학습합니다.

이번 프로젝트의 핵심 목표는 다음과 같습니다.

- Object Detection 기본 개념 이해
- pretrained Faster R-CNN 모델 불러오기
- 이미지 파일을 Tensor로 변환하기
- 객체 탐지 결과인 `boxes`, `labels`, `scores` 해석하기
- threshold를 적용하여 신뢰도 낮은 예측 제거하기
- bounding box를 이미지 위에 시각화하기

## Object Detection

이미지 분류는 이미지 전체에 대해 하나의 클래스를 예측합니다.

```text
Image Classification
이미지 → class
```

반면 Object Detection은 이미지 안의 객체 위치와 클래스를 함께 예측합니다.

```text
Object Detection
이미지 → bounding box + class label + confidence score
```

즉, Object Detection은 이미지 안에 어떤 객체가 있는지뿐만 아니라, 그 객체가 어디에 있는지도 예측합니다.

## Model

이번 프로젝트에서는 torchvision에서 제공하는 pretrained Faster R-CNN 모델을 사용했습니다.

```python
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
```

이 모델은 COCO 데이터셋 기준으로 학습된 객체 탐지 모델입니다.

이번 프로젝트에서는 모델을 직접 학습하지 않고, 이미 학습된 pretrained 모델을 사용하여 이미지 한 장에 대해 객체 탐지를 수행했습니다.

## Input Image

실행하려면 `object_detection_basic.py` 파일과 같은 폴더에 `sample.jpg` 이미지를 넣어야 합니다.

```text
13_object_detection_basic/
├── object_detection_basic.py
├── sample.jpg
└── README.md
```

Colab에서 실행할 경우에는 다음과 같이 예제 이미지를 다운로드할 수 있습니다.

```python
!wget -O sample.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
```

Colab에서는 이미지 경로를 다음과 같이 사용할 수 있습니다.

```python
image_path = "/content/sample.jpg"
```

GitHub에 정리한 `.py` 파일에서는 코드 파일이 있는 폴더 안의 `sample.jpg`를 찾도록 작성했습니다.

```python
image_path = Path(__file__).parent / "sample.jpg"
```

## Image Preprocessing

이미지는 PIL로 불러온 뒤 RGB 형식으로 변환했습니다.

```python
image = Image.open(image_path).convert("RGB")
```

그 후 `transforms.ToTensor()`를 사용하여 PyTorch Tensor로 변환했습니다.

```python
transform = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = transform(image).to(device)
```

입력 이미지 Tensor의 shape는 다음과 같이 출력되었습니다.

```text
torch.Size([3, 1213, 1546])
```

의미는 다음과 같습니다.

```text
3    → RGB 채널
1213 → 이미지 세로 크기
1546 → 이미지 가로 크기
```

## Object Detection Inference

Faster R-CNN 모델은 이미지 Tensor를 리스트 형태로 입력받습니다.

```python
with torch.no_grad():
    prediction = model([image_tensor])
```

이미지 한 장만 넣더라도 다음과 같이 리스트 형태로 입력해야 합니다.

```python
model([image_tensor])
```

`torch.no_grad()`는 예측 과정에서 gradient 계산을 하지 않도록 하는 코드입니다.

이번 프로젝트는 학습이 아니라 예측이 목적이므로 `torch.no_grad()`를 사용했습니다.

## Prediction Output

Faster R-CNN의 예측 결과는 다음 세 가지로 확인할 수 있습니다.

```python
boxes = prediction[0]["boxes"]
labels = prediction[0]["labels"]
scores = prediction[0]["scores"]
```

각각의 의미는 다음과 같습니다.

| Output | Meaning |
|---|---|
| `boxes` | 객체 위치 좌표 |
| `labels` | 예측한 클래스 번호 |
| `scores` | 예측 신뢰도 |

출력 shape는 다음과 같았습니다.

```text
boxes shape: torch.Size([4, 4])
labels shape: torch.Size([4])
scores shape: torch.Size([4])
```

이는 모델이 4개의 객체 후보를 탐지했다는 뜻입니다.

`boxes`의 shape가 `[4, 4]`인 이유는 4개의 객체 후보 각각이 `[x1, y1, x2, y2]` 좌표 4개를 가지기 때문입니다.

## Bounding Box

Bounding box는 객체 위치를 나타내는 사각형 좌표입니다.

```text
[x1, y1, x2, y2]
```

각 좌표의 의미는 다음과 같습니다.

```text
x1, y1 → 박스의 왼쪽 위 좌표
x2, y2 → 박스의 오른쪽 아래 좌표
```

예를 들어 다음 box는:

```text
[137.84, 67.79, 1386.90, 1172.82]
```

이미지 안에서 객체가 위치한 영역을 의미합니다.

## Threshold

Object Detection 모델은 여러 객체 후보를 출력할 수 있습니다.

하지만 모든 후보가 정확한 것은 아니기 때문에 `score`를 기준으로 필터링합니다.

```python
threshold = 0.7
```

이번 프로젝트에서는 score가 0.7 이상인 예측만 사용했습니다.

```python
if score.item() >= threshold:
```

## Experiment Result

이번 실험에서는 강아지 이미지 한 장을 입력으로 사용했습니다.

모델의 Top predictions는 다음과 같았습니다.

```text
1. class: dog, score: 0.9669
2. class: cat, score: 0.3522
3. class: frisbee, score: 0.3132
4. class: dog, score: 0.1133
```

threshold를 0.7로 설정했기 때문에 최종적으로는 `dog` 예측 하나만 사용했습니다.

```text
Prediction with score >= 0.7
class: dog, score: 0.9669
```

## Visualization

`matplotlib.patches.Rectangle`을 사용하여 이미지 위에 bounding box를 그렸습니다.

```python
rect = patches.Rectangle(
    (x1, y1),
    x2 - x1,
    y2 - y1,
    linewidth=2,
    edgecolor="red",
    facecolor="none"
)
```

여기서:

```text
(x1, y1) → 사각형의 시작점
x2 - x1 → 박스의 너비
y2 - y1 → 박스의 높이
```

`ax.text()`를 사용하여 bounding box 위에 class name과 score를 표시했습니다.

```python
ax.text(
    x1,
    y1,
    f"{class_name}: {score.item():.2f}",
    fontsize=12,
    color="white",
    bbox=dict(facecolor="red", alpha=0.5)
)
```

최종적으로 이미지 위에 다음과 같은 결과가 표시되었습니다.

```text
dog: 0.97
```

## What I Learned

이 프로젝트를 통해 다음 내용을 학습했습니다.

- Object Detection은 객체의 위치와 클래스를 함께 예측한다.
- 이미지 분류는 이미지 전체에 대해 하나의 클래스를 예측한다.
- Object Detection 모델은 `boxes`, `labels`, `scores`를 출력한다.
- `boxes`는 bounding box 좌표이다.
- `labels`는 예측한 클래스 번호이다.
- `scores`는 예측 신뢰도이다.
- threshold를 사용하여 신뢰도가 낮은 예측을 제거할 수 있다.
- Faster R-CNN은 이미지 Tensor를 리스트 형태로 입력받는다.
- `matplotlib.patches.Rectangle`로 bounding box를 시각화할 수 있다.

## Key Concepts

### Object Detection

Object Detection은 이미지 안의 객체 위치와 클래스를 함께 예측하는 작업입니다.

### Bounding Box

Bounding box는 객체가 이미지 안에서 어디에 있는지를 나타내는 사각형 좌표입니다.

형태는 `[x1, y1, x2, y2]`입니다.

### Label

Label은 모델이 예측한 객체의 클래스 번호입니다.

COCO class names 리스트를 사용하여 label 번호를 실제 클래스 이름으로 변환할 수 있습니다.

### Score

Score는 모델이 해당 예측을 얼마나 확신하는지를 나타내는 값입니다.

### Threshold

Threshold는 score가 일정 기준 이상인 예측만 사용하기 위한 기준값입니다.

이번 프로젝트에서는 `threshold = 0.7`을 사용했습니다.

### Faster R-CNN

Faster R-CNN은 객체 탐지 모델입니다.

이미지를 입력받아 객체의 위치, 클래스, 신뢰도를 예측합니다.

## Limitations

이번 프로젝트는 pretrained Faster R-CNN 모델을 사용하여 객체 탐지 결과를 확인하는 기초 실습입니다.

직접 객체 탐지 모델을 학습한 것은 아니며, 이미지 한 장에 대해 inference를 수행했습니다.

개선할 수 있는 부분은 다음과 같습니다.

- 여러 이미지에 대해 객체 탐지 수행
- threshold 값 변경에 따른 결과 비교
- 탐지 결과 이미지를 파일로 저장
- 다른 pretrained detection model 사용
- 직접 객체 탐지 데이터셋으로 학습 실습

## Next Step

다음 단계에서는 여러 이미지에 대해 객체 탐지를 수행하거나, detection 결과를 파일로 저장하는 기능을 추가할 수 있습니다.

이후에는 Object Detection 데이터셋 구조와 annotation 형식을 학습하고, 직접 모델을 학습하는 프로젝트로 확장할 수 있습니다.
