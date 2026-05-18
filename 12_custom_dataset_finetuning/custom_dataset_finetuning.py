# import + device
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Transform 설정
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

# Custom Dataset 불러오기
train_dir = "dataset/train"
val_dir = "dataset/val"

train_data = datasets.ImageFolder(
    root=train_dir,
    transform=train_transform
)

val_data = datasets.ImageFolder(
    root=val_dir,
    transform=val_transform
)

class_names = train_data.classes

print(class_names)
print(train_data.class_to_idx)
print(len(train_data))
print(len(val_data))

# DataLoader 만들기
batch_size = 32

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

val_dataloader = DataLoader(
    val_data,
    batch_size=batch_size,
    shuffle=False
)

# pretrained ResNet-18 모델 불러오기
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 전체 파라미터 먼저 고정
for param in model.parameters():
    param.requires_grad = False

# layer4는 다시 학습 가능하게 설정
for param in model.layer4.parameters():
    param.requires_grad = True

# Custom Dataset 클래스 수에 맞게 fc layer 수정
num_features = model.fc.in_features
num_classes = len(class_names)
model.fc = nn.Linear(num_features, num_classes)

model = model.to(device)
print(model)

# loss 함수와 optimizer 설정
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4
)

# train 함수 작성
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

# validation 함수
def validate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    val_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size

    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

# 학습 실행 코드
epochs = 1

for t in range(epochs):
    print(f"Epoch {t + 1}\n--------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    validate(val_dataloader, model, loss_fn)

print("Done!")

# 모델 저장 코드
torch.save(model.state_dict(), "custom_dataset_finetuning.pth")
print("Saved PyTorch Model State to custom_dataset_finetuning.pth")
