# import + device
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


# pretrained Faster R-CNN 모델 불러오기
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)

model = model.to(device)
model.eval()

print("Model loaded!")


# 이미지 불러오기
image_path = Path(__file__).parent / "sample.jpg"

image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor()
])

image_tensor = transform(image).to(device)

print(type(image))
print(image_tensor.shape)


# 객체 탐지 실행
with torch.no_grad():
    prediction = model([image_tensor])


# 예측 결과 확인
boxes = prediction[0]["boxes"]
labels = prediction[0]["labels"]
scores = prediction[0]["scores"]

print("boxes shape:", boxes.shape)
print("labels shape:", labels.shape)
print("scores shape:", scores.shape)

print("\nTop 5 predictions")

for i in range(min(5, len(scores))):
    label_id = labels[i].item()
    class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
    score = scores[i].item()
    box = boxes[i].tolist()

    print(f"{i + 1}. class: {class_name}, score: {score:.4f}, box: {box}")


# threshold 적용
threshold = 0.7

print(f"\nPrediction with score >= {threshold}")

for i in range(len(scores)):
    if scores[i].item() >= threshold:
        label_id = labels[i].item()
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
        score = scores[i].item()
        box = boxes[i].tolist()

        print(f"class: {class_name}, score: {score:.4f}, box: {box}")


# 객체 탐지 결과 시각화
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

for box, label, score in zip(boxes.cpu(), labels.cpu(), scores.cpu()):
    if score.item() >= threshold:
        x1, y1, x2, y2 = box.tolist()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )

        ax.add_patch(rect)

        label_id = label.item()
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]

        ax.text(
            x1,
            y1,
            f"{class_name}: {score.item():.2f}",
            fontsize=12,
            color="white",
            bbox=dict(facecolor="red", alpha=0.5)
        )

plt.axis("off")
plt.show()
