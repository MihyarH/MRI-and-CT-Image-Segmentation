import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import UNet3D
from torch.utils.data import DataLoader
from dataset import MedicalImageDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
image_dir = "../data/amos22/imagesTr"
label_dir = "../data/amos22/labelsTr"

# Initialize dataset and dataloader with pin_memory for faster data transfer
dataset = MedicalImageDataset(image_dir=image_dir, label_dir=label_dir)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True)

# Initialize model, loss, and optimizer
model = UNet3D(in_channels=1, out_channels=1).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dice Score function with no_grad to save memory
def dice_score(preds, labels, threshold=0.5):
    with torch.no_grad():
        preds = (preds > threshold).float()
        intersection = (preds * labels).sum()
        union = preds.sum() + labels.sum()
        dice = (2. * intersection) / (union + 1e-6)  # Small epsilon to avoid division by zero
    return dice.item()

# Training loop with memory management and saving model after each epoch
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="3d_unet_model.pth"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_dice = 0.0
        num_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # Move data to GPU and set gradients to zero
            images = images.float().to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)
            optimizer.zero_grad()

            # Forward pass and compute loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update loss and dice score
            running_loss += loss.item() * images.size(0)
            running_dice += dice_score(outputs, labels)
            num_batches += 1

            # Clear memory
            del images, labels, outputs, loss
            torch.cuda.empty_cache()  # Clears GPU cache

        # Calculate epoch loss and dice score
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = running_dice / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Dice Score: {epoch_dice:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), f"{save_path}_epoch{epoch + 1}.pth")
        print(f"Model saved to {save_path}_epoch{epoch + 1}.pth")

    print("Training completed.")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path="3d_unet_model")
