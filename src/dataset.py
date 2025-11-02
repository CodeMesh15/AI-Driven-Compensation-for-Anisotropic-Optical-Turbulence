# src/dataset.py
# PyTorch Dataset class for loading the turbulence data.

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TurbulenceDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        """
        Args:
            image_dir (string): Directory with all the distorted beam images.
            labels_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_file)
        self.transform = transform
        
        # Define a default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Resize for ResNet
                transforms.ToTensor(),
                # Normalize for 1 channel
                transforms.Normalize([0.5], [0.5]) 
            ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Load image
        img_filename = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_filename)
        # Open as grayscale (L) to match our 1-channel model
        image = Image.open(img_path).convert('L') 

        # 2. Get labels
        # Labels start from the 2nd column (index 1)
        labels = self.labels_df.iloc[idx, 1:].values
        labels = labels.astype('float').reshape(-1) # Ensure 1D float array
        
        # 3. Apply transform
        if self.transform:
            image = self.transform(image)
            
        # 4. Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels
