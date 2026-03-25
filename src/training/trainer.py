import os
import torch
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, checkpoint_dir, patience=10):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Scheduler remains active to "fine-tune" the weights during long runs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float('inf')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        epoch_loss, correct_preds, total_preds = 0.0, 0, 0
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct_preds += torch.sum(predictions == y_batch).item()
            total_preds += y_batch.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return epoch_loss / total_preds, correct_preds / total_preds

    def validate_epoch(self):
        self.model.eval()
        epoch_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                epoch_loss += loss.item() * X_batch.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct_preds += torch.sum(predictions == y_batch).item()
                total_preds += y_batch.size(0)
                
        return epoch_loss / total_preds, correct_preds / total_preds

    def fit(self, epochs):
        """Modified: Early Stopping is disabled. The loop runs for the full 'epochs' count."""
        print(f"[System] Starting DEEP CONVERGENCE training on {self.device} for {epochs} epochs...")
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch:03d}/{epochs} | "
                  f"Loss [T/V]: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc [T/V]: {train_acc:.4f}/{val_acc:.4f} | LR: {current_lr:.6f}")
            
            # --- Always Checkpoint the Best Model ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0 
                save_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"   -> [Save] Found new global minimum. Weights updated.")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    # We print a warning, but we DON'T 'break' the loop
                    print(f"   -> [Monitor] No improvement for {self.counter} epochs. Continuing deep convergence...")
                
        return history