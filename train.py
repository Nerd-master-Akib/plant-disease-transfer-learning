import torch 
import torch.nn as nn
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split 
from models.model import PlantResNet18 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    root = "D:/PlantDisease_TransferLearning/A CNN Project/data/train/plantvillage dataset/color",
    transform = transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset)) 

from torch.utils.data import random_split
train_size = int(0.7*len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(
    dataset, [train_size, val_size, test_size]
)

print("train:", len(train_data))
print("validation:", len(val_data))
print("test:", len(test_data))


from torch.utils.data import DataLoader


train_loader = DataLoader(
    train_data,
    batch_size = 16,
    shuffle = True
)


val_loader = DataLoader(
    val_data,
    batch_size = 16,
    shuffle = False
)

test_loader = DataLoader(
    test_data,
    batch_size = 16,
    shuffle = False
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

print(dataset.class_to_idx) 

## First CNN codes: 


    
model = PlantResNet18().to(device)
print(model)

sample_batch, _ = next(iter(train_loader))
sample_batch = sample_batch.to(device)

output = model(sample_batch)

print(output.shape)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.model.fc.parameters(),
    lr=0.001
)

## First Training Loop

epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images,labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct/total
        print(
        f"Epoch {epoch+1}/{epochs}, "
        f"Loss: {running_loss:.4f}"
        f"Accuracy: {accuracy:.2f}%"
    )
        
torch.save(model.state_dict(), "plant_resnet18.pth")
print("Model saved!")     
        
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images,labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

model.load_state_dict(torch.load("plant_resnet18.pth"))
model.eval()

def predict_image(image,model):
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output,1)
        return predicted 