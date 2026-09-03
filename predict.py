import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class PlantCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            
            
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128*28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 5)) 
        
    def forward(self,x):
        x = self.conv_layers(x)
        
        x = torch.flatten(x, start_dim = 1)
        
        x = self.fc_layers(x)
        
        return x
    
model = PlantCNN()
model.load_state_dict(torch.load('plant_cnn.pth'))
model.eval()
print("Model loaded successfully.")

class_names = [
    "Potato Early Blight",
    "Potato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Healthy"
]


from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open("Test_leaf.jpg").convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(output, 1)

predicted_class = class_names[predicted.item()]

print("Prediction:", predicted_class)