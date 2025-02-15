import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load trained model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

# Example usage
# embedding = extract_features("./img/db/image1.png")
# np.save("./img/db/image1_embedding.npy", embedding)  # Save features
import os

def process_all_images(db_folder="./img/db"):
    # Create output directory if it doesn't exist
    os.makedirs(db_folder, exist_ok=True)
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(db_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(db_folder, image_file)
        # Get filename without extension
        base_name = os.path.splitext(image_file)[0]
        
        try:
            # Extract features
            embedding = extract_features(image_path)
            # Save embedding with same name as image but with .npy extension
            output_path = os.path.join(db_folder, f"{base_name}_embedding.npy")
            np.save(output_path, embedding)
            print(f"Processed: {image_file}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

# Example usage
process_all_images()