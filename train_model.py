import torch
import torchvision.models as models
import torch.nn as nn

# Load ResNet50 (pretrained)
model = models.resnet50(pretrained=True)

# Replace the last layer with a custom classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10 classes for cava brands

# Save the model
torch.save(model.state_dict(), "models/cava_model.pth")
