# import + device
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt

device = "cuda" if  torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Transform 설정
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

# CIFAR-10 데이터셋 불러오기
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=train_transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform
)

# DataLoader 만들기
batch_size = 32

train_dataloader = DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
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

# 마지막 fc layer 수정
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

model = model.to(device)
print(model)

# loss 함수와 optimizer 설정
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(model.layer4.parameters()) + list(model.fc.parameters()),
    lr=1e-4
)

# train 함수
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

        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

# test 함수
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 학습 실행 코드
epochs = 1

for t in range(epochs):
    print(f"Epoch {t + 1}\n-----------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# 모델 저장 코드
torch.save(model.state_dict(), "finetuning_resnet18.pth")
print("Saved PyTorch Model State to finetuning_resnet18.pth")
