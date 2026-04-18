import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import pandas as pd 

from power_dataset import CircuitPowerDataset
from gnn_model import BullsEyePredictor # Updated to match your new architecture

# --- CHECKPOINT PATH ---
CHECKPOINT_PATH = "/content/drive/MyDrive/gnn_checkpoint.pth"
BEST_MODEL_PATH = "/content/drive/MyDrive/best_model.pth"

# ==========================================
# THE EARLY STOPPING ALGORITHM
# ==========================================
class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, epoch):
        """Saves model when validation loss decreases."""
        print(f"🌟 Validation loss decreased to {val_loss:.6f}. Saving best model...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Keeping these keys at 1.0 so older scripts unpacking the dictionary don't crash
            'max_power': 1.0,
            'max_area': 1.0,
            'max_delay': 1.0
        }, BEST_MODEL_PATH)


def train_model():
    print("🚀 Setting up the Production Pipeline with Auto-Resume & Early Stopping...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Accelerator Activated: {device.type.upper()}\n")

    dataset = CircuitPowerDataset(csv_file="dataset/labels.csv", zip_path="dataset.zip")
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = BullsEyePredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Matched IEEE paper learning rate
    
    epochs = 100 # Can set high because Early Stopping will catch it
    start_epoch = 0 
    
    # Initialize Early Stopping (Will stop if no improvement for 7 epochs)
    early_stopping = EarlyStopping(patience=7, delta=0.0001)
    
    # ==========================================
    #      THE AUTO-RESUME LOGIC
    # ==========================================
    if os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Snapshot found at {CHECKPOINT_PATH}!")
        print("Restoring brain states to resume training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Successfully resumed! Restarting from Epoch {start_epoch + 1}...\n")
    else:
        print("\nNo previous snapshot found. Starting fresh from Epoch 1...\n")

    print(f"Training on {train_size} scenarios | Validating on {val_size} scenarios")
    
    for epoch in range(start_epoch, epochs):
        # --- PHASE 1: TRAINING ---
        model.train() 
        total_train_loss = 0
        
        for batch_idx, (graphs, recipes) in enumerate(train_loader):
            graphs = graphs.to(device)
            recipes = recipes.to(device)
            
            optimizer.zero_grad()
            predictions = model(graphs.x, graphs.edge_index, graphs.batch, recipes)
            
            # Predictions
            pred_power = predictions[:, 0]
            pred_area = predictions[:, 1]
            pred_delay = predictions[:, 2]
            
            # True Targets (Already Intensive/Per-Node from Dataset)
            true_power = graphs.y_power.squeeze()
            true_area = graphs.y_area.squeeze()
            true_delay = graphs.y_delay.squeeze()
            
            # Loss Calculation (No double-dipping normalization needed)
            loss_power = criterion(pred_power, true_power)
            loss_area = criterion(pred_area, true_area)
            loss_delay = criterion(pred_delay, true_delay)
            
            batch_loss = (1.0 * loss_power) + (0.1 * loss_area) + (0.1 * loss_delay)
            
            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- PHASE 2: VALIDATION ---
        model.eval() 
        total_val_loss = 0
        
        with torch.no_grad():
            for graphs, recipes in val_loader:
                graphs = graphs.to(device)
                recipes = recipes.to(device)
                
                predictions = model(graphs.x, graphs.edge_index, graphs.batch, recipes)
                
                pred_power = predictions[:, 0]
                pred_area = predictions[:, 1]
                pred_delay = predictions[:, 2]
                
                true_power = graphs.y_power.squeeze()
                true_area = graphs.y_area.squeeze()
                true_delay = graphs.y_delay.squeeze()
                
                loss_power = criterion(pred_power, true_power)
                loss_area = criterion(pred_area, true_area)
                loss_delay = criterion(pred_delay, true_delay)
                
                batch_loss = (1.0 * loss_power) + (0.1 * loss_area) + (0.1 * loss_delay)
                total_val_loss += batch_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save standard recovery snapshot (in case Google Colab crashes mid-epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

        # Trigger Early Stopping Check
        early_stopping(avg_val_loss, model, optimizer, epoch)
        
        if early_stopping.early_stop:
            print("\n🛑 EARLY STOPPING TRIGGERED. The model has reached its maximum generalization potential.")
            break

    print("\n✅ Production Training Complete! The absolute best brain is saved as 'best_model.pth'.")

if __name__ == "__main__":
    train_model()