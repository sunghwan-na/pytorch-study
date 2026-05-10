#import+device+데이터 준비
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#Transform 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#CIFAR-10 데이터셋 불러오기
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

#DataLoader 만들기
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

#CNN 모델 클래스 작성
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        logits =self.classifier(x)
        return logits
    
model = CNN().to(device)
print(model)
    
#loss 함수와 optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

#train 함수 작성
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
            current = batch*len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

#test 함수 작성
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#학습 실행 코드
epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")

#모델 저장 코드
torch.save(model.state_dict(), "cifar10_cnn.pth")
print("Saved PyTorch Model State to cifar10_cnn.pth")

#예측 이미지 시각화
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

        #Normalize를 되돌려서 이미지가 자연스럽게 보이도록 변환
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
