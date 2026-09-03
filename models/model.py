import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PlantResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT) 
        
        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            5
        )
        
    def forward(self, x):
            return self.model(x)
            
            
