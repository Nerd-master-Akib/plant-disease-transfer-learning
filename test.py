import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import PlantResNet18


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load the dataset
dataset = datasets.ImageFolder(
    root="D:/PlantDisease_TransferLearning/A CNN Project/data/train/plantvillage dataset/color",
    transform=transform
)


# IMPORTANT:
# We need the SAME train/validation/test split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size]
)


test_loader = DataLoader(
    test_data,
    batch_size=16,
    shuffle=False
)


# Create the model
model = PlantResNet18().to(device)


# Load the trained weights
model.load_state_dict(
    torch.load(
        "plant_resnet18.pth",
        map_location=device
    )
)

model.eval()


# Test evaluation
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")