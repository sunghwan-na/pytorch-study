# import + device
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Transform 설정
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
batch_size = 64

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

# ResNet BasicBlock 만들기
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_layer(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)
        return out
    
# ResNet-18 style 모델 만들기
class ResNet18CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_channels = 32

        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, num_blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []

        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        logits = self.classifier(x)
        return logits
    
# 모델 생성
model = ResNet18CIFAR10().to(device)
print(model)

# loss 함수와 optimizer 설정
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

#train 함수
def train(dataloader, model, loss_fn, optimizer):
    size =  len(dataloader.dataset)
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

#test 함수
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

# 학습 실행
epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n--------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

# 모델 저장
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Saved PyTorch Model State to resnet18_cifar10.pth")

# 예측 이미지 시각화
classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

model.eval()

figure = plt.figure(figsize=(10, 8))
cols, rows = 3, 3

with torch.no_grad():
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = test_data[sample_idx]

        X = img.unsqueeze(0).to(device)

        pred = model(X)
        predicted_label = pred.argmax(1).item()

        # Normalize를 되돌려서 이미지가 자연스럽게 보이도록 변환
        img_for_show = img / 2 + 0.5
        img_for_show = img_for_show.permute(1, 2, 0)

        figure.add_subplot(rows, cols, i)
        plt.title(
            f"Pred: {classes[predicted_label]}\nTrue: {classes[label]}",
            fontsize=9
        )
        plt.axis("off")
        plt.imshow(img_for_show)

plt.tight_layout()
plt.show()
