import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import ChestXrayModel

def train_model(train_dir, val_dir, epochs=5, batch_size=32, lr=0.001):
    # 1. Hardware Selection: Automatically use GPU if available (Kaggle), else fallback to CPU (local testing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data Pipeline
    print("Initializing DataLoaders...")
    train_loader, val_loader, classes = get_dataloaders(train_dir, val_dir, batch_size=batch_size)

    # 3. Initialize Model and move its parameters to the selected device (CPU/GPU)
    model = ChestXrayModel(pretrained=True).to(device)

    # 4. Define Loss Function and Optimizer
    # BCEWithLogitsLoss is mathematically optimized for binary classification (Normal vs Pneumonia)
    criterion = nn.BCEWithLogitsLoss()
    
    # We only want to train the final classifier layer parameters to save time and preserve pre-trained features
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=lr)

    # Loop through the training cycles (Epochs)
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # ================= TRAINING PHASE =================
        model.train() # Tell the model it is in training mode (enables data augmentation/dropout)
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Move data tensors to the identical device as the model
            images, labels = images.to(device), labels.to(device)
            
            # Clear previous gradients from the last step
            optimizer.zero_grad()
            
            # Forward Pass: Compute raw model outputs (logits)
            outputs = model(images).squeeze(1) # Squeeze converts shape [batch_size, 1] to [batch_size]
            
            # Calculate Loss
            loss = criterion(outputs, labels)
            
            # Backward Pass: Compute gradients
            loss.backward()
            
            # Step Optimizer: Update model weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * images.size(0)
            
            # Convert raw logits to probabilities (0 to 1) using Sigmoid for metrics reporting
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # ================= VALIDATION PHASE =================
        model.eval() # Tell the model to turn off training configurations (disables data augmentation)
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # Disabling gradient calculation speeds up computation and saves memory during evaluation
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    # 5. Save the trained model parameters to disk
    # This weights file is the ONLY thing we need to pull back down to our local Streamlit app
    torch.save(model.state_dict(), "model.pth")
    print("\nTraining Complete! Model weights successfully saved as 'model.pth'")

if __name__ == "__main__":
    # Point these paths to your local dummy folder setup to verify the script runs without errors
    # Once verified, we will copy this code directly into Kaggle.
    train_data_path = "./data/train"
    val_data_path = "./data/val"
    
    # Run a quick 1-epoch test locally with dummy files
    train_model(train_dir=train_data_path, val_dir=val_data_path, epochs=1, batch_size=2)