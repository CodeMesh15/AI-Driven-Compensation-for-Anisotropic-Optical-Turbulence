# Script to load a trained model and make a prediction on a single image.

import torch
import pandas as pd
from PIL import Image
import argparse
import os
from torchvision import transforms

from src.model import CompensationNet
# We need the transform from the dataset
from src.dataset import TurbulenceDataset 

def predict(model, image_path, transform):
    """Makes a prediction on a single image."""
    
    image = Image.open(image_path).convert('L') # Open as Grayscale
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
        
    return prediction.cpu().numpy().flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict turbulence compensation")
    parser.add_argument('--image', type=str, required=True, help='Path to the distorted beam image.')
    parser.add_argument('--model', type=str, default='models/compensation_model.pth', help='Path to the trained model file.')
    args = parser.parse_args()

    # 1. Get the data transform
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 2. Get number of params model was trained on
    # We read the header of the labels file
    try:
        labels_df = pd.read_csv(os.path.join('data', 'labels.csv'))
        num_output_params = len(labels_df.columns) - 1
    except FileNotFoundError:
        print("Warning: 'data/labels.csv' not found. Assuming 14 output parameters.")
        num_output_params = 14 # Default (j=2 to j=15)

    # 3. Load the model
    model = CompensationNet(num_output_params=num_output_params)
    try:
        model.load_state_dict(torch.load(args.model))
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model}")
        print("Please run train.py first to train and save a model.")
        exit()

    # 4. Make prediction
    try:
        predicted_coeffs = predict(model, args.image, data_transform)
        print("Prediction Complete.")
        print(f"Image: {args.Tmage}")
        print(f"Predicted Zernike Coefficients (j=2 to j={num_output_params+1}):")
        
        coeffs_str = [f"{c:.4f}" for c in predicted_coeffs]
        print(", ".join(coeffs_str))
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image}")
