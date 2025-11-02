# Main script to train the CompensationNet model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Import our custom modules
from src.dataset import TurbulenceDataset
from src.model import CompensationNet

# --- Configuration ---
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
MODEL_SAVE_PATH = "models/compensation_model.pth"

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20 # Set to 2-3 for a quick test, 20+ for real training
VALID_SPLIT = 0.2 # 20% of data for validation

def train():
    """Main training function."""
    
    # 1. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Define transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize for ResNet
        transforms.ToTensor(),
        # Normalize for 1 channel (mean=0.5, std=0.5)
        transforms.Normalize([0.5], [0.5]) 
    ])

    # 3. Create dataset and dataloaders
    print("Loading dataset...")
    full_dataset = TurbulenceDataset(
        image_dir=IMAGE_DIR, 
        labels_file=LABELS_FILE, 
        transform=data_transform
    )
    
    # Determine number of output parameters from the dataset
    # (N_cols - 1 for filename)
    num_output_params = len(full_dataset.labels_df.columns) - 1
    print(f"Model will predict {num_output_params} parameters.")

    # Split into training and validation sets
    val_size = int(len(full_dataset) * VALID_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("Dataset loaded and split.")

    # 4. Initialize model, loss, and optimizer
    model = CompensationNet(num_output_params=num_output_params).to(device)
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        history['train_loss'].append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ... New best model saved to {MODEL_SAVE_PATH}")

    print("Finished Training.")
    
    # 6. Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Training loss curve saved to 'training_loss_curve.png'")

if __name__ == "__main__":
    train()
