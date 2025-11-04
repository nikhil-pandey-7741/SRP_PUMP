import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import SparseVGGNet


class DynamographDataset(Dataset):
    """Dataset for training (NO augmentation - dataset already augmented)"""
    
    def __init__(self, csv_file, image_column, label_columns, base_path='', 
                 image_size=(190, 400)):
        self.df = pd.read_csv(csv_file)
        self.image_column = image_column
        self.label_columns = label_columns
        self.base_path = base_path
        self.image_size = image_size
        
        # Simple transform without augmentation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.df.iloc[idx][self.image_column])
        
        try:
            image = Image.open(img_path).convert('L')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(1, *self.image_size)
        
        labels = self.df.iloc[idx][self.label_columns].values.astype(np.float32)
        label = np.argmax(labels)
        
        return image, label


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


def calibrate_temperature(model, val_loader, device, max_iter=50):
    """Calibrate temperature parameter on validation set"""
    print("\n" + "="*70)
    print("CALIBRATING TEMPERATURE ON VALIDATION SET")
    print("="*70)
    
    model.eval()
    
    # Collect all logits and labels
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collecting logits"):
            images = images.to(device)
            outputs, _, _ = model(images)
            # Remove temperature scaling temporarily
            outputs = outputs * model.temperature
            logits_list.append(outputs)
            labels_list.append(labels.to(device))
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Optimize temperature
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / model.temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    temperature_value = model.temperature.item()
    print(f"\n✓ Optimal temperature: {temperature_value:.4f}")
    
    return temperature_value


def train_cnn(train_csv, val_csv, test_csv, base_path, label_columns,
              num_epochs=100, batch_size=32, learning_rate=0.001,
              save_dir='./trained_models'):
    """
    Train SparseVGGNet with all enhancements
    """
    
    print("\n" + "="*70)
    print("TRAINING SPARSEVGGNET WITH ENHANCEMENTS")
    print("="*70)
    print("\nEnhancements:")
    print("  ✓ Reduced Dropout (0.3)")
    print("  ✓ Label Smoothing (0.1)")
    print("  ✓ Cosine Annealing LR Scheduler")
    print("  ✓ Temperature Scaling")
    print("  ✓ Early Stopping")
    print("="*70 + "\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_dataset = DynamographDataset(
        train_csv, 'image_path', label_columns, base_path, 
        image_size=(190, 400)
    )
    val_dataset = DynamographDataset(
        val_csv, 'image_path', label_columns, base_path,
        image_size=(190, 400)
    )
    test_dataset = DynamographDataset(
        test_csv, 'image_path', label_columns, base_path,
        image_size=(190, 400)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}\n")
    
    # Initialize model
    model = SparseVGGNet(num_classes=len(label_columns), sparse=True)
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine Annealing LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print("Starting training...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs, _, _ = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 70)
    
    # Load best model for calibration
    print("\n" + "="*70)
    print("Loading best model for temperature calibration...")
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calibrate temperature
    calibrate_temperature(model, val_loader, device)
    
    # Save calibrated model
    torch.save({
        'epoch': checkpoint['epoch'],
        'model_state_dict': model.state_dict(),
        'train_acc': checkpoint['train_acc'],
        'val_acc': checkpoint['val_acc'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
        'temperature': model.temperature.item()
    }, os.path.join(save_dir, 'best_model.pth'))
    
    print(f"✓ Calibrated model saved to {save_dir}/best_model.pth")
    
    # Test evaluation
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f"\n✓ Test Accuracy: {test_acc:.2f}%\n")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"✓ Training curves saved to {save_dir}/training_curves.png\n")
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }
