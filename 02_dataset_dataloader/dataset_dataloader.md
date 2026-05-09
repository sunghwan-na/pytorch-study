# PyTorch 02 - Dataset과 DataLoader

## 1. Dataset이란?

`Dataset`은 데이터를 저장하고, 특정 index로 데이터에 접근할 수 있게 해주는 객체이다.

```python
img, label = training_data[index]
```

`training_data[index]`를 하면 데이터 하나가 나오고, 그 안에는 다음 두 가지가 들어있다.

```text
이미지 img + 정답 label
```

즉, Dataset은 이미지와 정답 label을 묶어서 관리하는 역할을 한다.

---

## 2. FashionMNIST 데이터셋 불러오기

FashionMNIST는 의류 이미지 데이터셋이다.

- 학습 데이터: 60,000개
- 테스트 데이터: 10,000개
- 이미지 크기: 28 × 28
- 이미지 형태: 흑백 이미지
- 분류 클래스: 10개

각 데이터는 이미지 데이터와 정답 label로 구성된다.

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

### 주요 매개변수

- `root="data"`  
  데이터가 저장될 폴더 경로를 의미한다.

- `train=True`  
  학습용 데이터를 불러온다.

- `train=False`  
  테스트용 데이터를 불러온다.

- `download=True`  
  `root` 경로에 데이터가 없으면 자동으로 다운로드한다.

- `transform=ToTensor()`  
  이미지 데이터를 PyTorch Tensor 형태로 변환한다.

---

## 3. Dataset 시각화하기

Dataset은 리스트처럼 index로 직접 접근할 수 있다.  
아래 코드는 FashionMNIST 학습 데이터에서 랜덤 이미지 9개를 뽑아 3×3 형태로 출력한다.

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]

    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
```

### 핵심 개념

- `labels_map`  
  숫자 label을 사람이 이해할 수 있는 옷 이름으로 바꿔준다.

- `torch.randint()`  
  학습 데이터 중 랜덤한 index를 뽑는다.

- `img, label = training_data[sample_idx]`  
  해당 index의 이미지와 label을 가져온다.

- `img.squeeze()`  
  FashionMNIST 이미지의 shape는 보통 `[1, 28, 28]`이다.  
  이미지를 화면에 출력할 때는 `[28, 28]` 형태가 필요하므로 `squeeze()`로 앞의 `1` 차원을 제거한다.

- `cmap="gray"`  
  이미지를 흑백으로 출력한다.

---

## 4. 파일에서 사용자 정의 Dataset 만들기

사용자 정의 Dataset은 PyTorch가 기본으로 제공하는 FashionMNIST 같은 데이터셋이 아니라,  
내가 직접 가진 이미지 파일과 라벨 CSV 파일을 이용해 Dataset을 만드는 것이다.

일반적인 구조는 다음과 같다.

```text
images/
├── tshirt1.jpg
├── tshirt2.jpg
└── ankleboot999.jpg

annotations.csv
tshirt1.jpg,0
tshirt2.jpg,0
ankleboot999.jpg,9
```

이미지는 `img_dir` 디렉토리에 저장되고,  
정답 label은 `annotations_file` CSV 파일에 별도로 저장된다.

```python
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=["file_name", "label"])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform  

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
```

### 사용자 정의 Dataset의 핵심 함수

사용자 정의 Dataset 클래스는 보통 3개의 함수를 구현한다.

#### 1. `__init__`

Dataset 객체가 생성될 때 한 번 실행된다.

역할:

- CSV 파일 읽기
- 이미지 폴더 경로 저장
- transform 저장
- target_transform 저장

```python
self.img_labels = pd.read_csv(annotations_file, names=["file_name", "label"])
```

CSV 파일을 읽어서 이미지 파일명과 label 정보를 저장한다.

```python
self.img_dir = img_dir
```

이미지가 들어있는 폴더 경로를 저장한다.

---

#### 2. `__len__`

```python
def __len__(self):
    return len(self.img_labels)
```

데이터셋의 전체 샘플 개수를 반환한다.

---

#### 3. `__getitem__`

```python
def __getitem__(self, idx):
```

주어진 index에 해당하는 이미지와 label을 불러와 반환한다.

핵심 흐름:

```text
1. CSV에서 idx번째 이미지 파일명 가져오기
2. 이미지 폴더 경로와 파일명을 합쳐 이미지 경로 만들기
3. read_image()로 이미지를 Tensor로 읽기
4. CSV에서 idx번째 label 가져오기
5. transform이 있으면 이미지에 적용
6. target_transform이 있으면 label에 적용
7. image, label 반환
```

---

## 5. DataLoader로 학습용 데이터 준비하기

`Dataset`은 데이터 하나씩 접근할 수 있게 해준다.  
하지만 모델 학습에서는 데이터를 보통 여러 개씩 묶은 mini-batch 단위로 사용한다.

`DataLoader`는 Dataset을 batch 단위로 묶어서 꺼내는 역할을 한다.

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
```

### 주요 매개변수

- `batch_size=64`  
  데이터를 64개씩 묶어서 가져온다.

- `shuffle=True`  
  데이터를 섞어서 가져온다.  
  학습 데이터는 일반적으로 섞어서 사용한다.

- `shuffle=False`  
  데이터를 섞지 않는다.  
  테스트 데이터는 보통 섞을 필요가 없다.

---

## 6. DataLoader 순회하기

DataLoader는 데이터를 batch 단위로 꺼낸다.

```python
train_features, train_labels = next(iter(train_dataloader))

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap="gray")
plt.show()

print(f"Label: {label}")
```

출력 예시:

```text
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 1
```

### 출력 shape 의미

```text
train_features shape = [64, 1, 28, 28]
```

의미:

- `64`: 이미지 64개
- `1`: 흑백 채널
- `28`: 이미지 세로
- `28`: 이미지 가로

```text
train_labels shape = [64]
```

의미:

- 이미지 64개에 대한 정답 label 64개

---

## 정리

- `Dataset`은 데이터 하나씩 접근할 수 있게 해준다.
- `training_data[index]`를 하면 이미지와 label이 나온다.
- `FashionMNIST`는 PyTorch에서 제공하는 의류 이미지 데이터셋이다.
- `ToTensor()`는 이미지를 Tensor 형태로 변환한다.
- 사용자 정의 Dataset은 직접 가진 이미지와 CSV 라벨 파일로 Dataset을 만드는 방식이다.
- 사용자 정의 Dataset은 `__init__`, `__len__`, `__getitem__`을 구현한다.
- `DataLoader`는 Dataset을 batch 단위로 묶어서 꺼내준다.
- `batch_size=64`는 한 번에 64개의 데이터를 가져온다는 뜻이다.
- 학습 데이터는 보통 `shuffle=True`, 테스트 데이터는 보통 `shuffle=False`로 사용한다.

## Reference

- PyTorch Tutorials: https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html
