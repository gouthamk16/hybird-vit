import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from vit import HybridVisionTransformer
import gc

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # Set to False for better performance
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance

# Custom dataset class for cat images
class CatDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, val_split=0.2):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # Get all class folders
        self.class_folders = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Create a mapping from folder name to class index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_folders)}
        
        # Get all image paths and their corresponding labels
        self.samples = []
        for class_folder in self.class_folders:
            folder_path = os.path.join(root_dir, class_folder)
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
            for img_path in images:
                self.samples.append((img_path, self.class_to_idx[class_folder]))
        
        # Split data into train and validation sets
        random.seed(42)  # For reproducible splits
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * (1 - val_split))
        if self.train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"{'Training' if train else 'Validation'} dataset contains {len(self.samples)} images across {len(self.class_folders)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ArcFace loss for better class separation in large-scale classification
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.threshold = torch.cos(torch.tensor(np.pi - margin))
        self.mm = torch.sin(torch.tensor(np.pi - margin)) * margin

    def forward(self, input, label):
        # Normalize weights and features
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # Calculate arcface logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding for the ground truth
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # Calculate output logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output, cosine * self.scale

# Training function with AMP (Automatic Mixed Precision)
def train_epoch(model, dataloader, criterion, optimizer, device, arcface=None, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            # Forward pass
            if arcface:
                logits, embeddings = model(images)
                outputs, _ = criterion(embeddings, labels)
            else:
                logits, embeddings = model(images)
                outputs = logits
                
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Optimizer step with gradient scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_description(f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        # Clear some memory
        del images, labels, outputs, embeddings
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device, arcface=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision for validation (optional but can be faster)
            with torch.cuda.amp.autocast():
                # Forward pass
                if arcface:
                    logits, embeddings = model(images)
                    outputs, _ = criterion(embeddings, labels)
                else:
                    logits, embeddings = model(images)
                    outputs = logits
                    
                loss = F.cross_entropy(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear some memory
            del images, labels, outputs, embeddings
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / total
    
    return val_loss, val_acc

# Main training loop
def main():
    parser = argparse.ArgumentParser(description='Train a cat classification model with Mixed Precision')
    parser.add_argument('--data_dir', type=str, default='train', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--use_arcface', action='store_true', help='Use ArcFace loss')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints_amp', help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use')
    parser.add_argument('--img_size', type=int, default=160, help='Image size for training')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Mixed precision training requires CUDA.")
        return
    
    # Set CUDA device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}')
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Print CUDA amp info
    print("Mixed precision training ENABLED by default")
    print("FP16 operations supported:", torch.cuda.is_bf16_supported())
    
    # Initialize mixed precision training (ALWAYS enabled in this script)
    scaler = torch.cuda.amp.GradScaler()
    
    # Data transformations optimized for faster training
    train_transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(args.img_size + 32),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = CatDataset(args.data_dir, transform=train_transform, train=True)
    val_dataset = CatDataset(args.data_dir, transform=val_transform, train=False)
    
    # Optimize performance with pin_memory and persistent workers
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for more stable training
        persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Calculate the number of classes from the dataset
    num_classes = len(train_dataset.class_folders)
    print(f"Training with {num_classes} classes")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize model with optimized settings
    model = HybridVisionTransformer(
        num_layers=8,
        emb_size=768,
        num_head=8,
        num_class=num_classes,
        img_size=args.img_size
    ).to(device)
    
    # Enable DataParallel for multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    # Initialize ArcFace loss if specified
    arcface = None
    if args.use_arcface:
        arcface = ArcFaceLoss(1024, num_classes).to(device)  # 1024 is the embedding dimension
        optimizer = optim.AdamW([
            {'params': model.parameters()},
            {'params': arcface.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Create a log file
    log_file = os.path.join(args.save_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Mixed Precision Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 50 + "\n")
    
    # Log initial memory usage
    print(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    
    # Training start time
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, arcface, optimizer, device, arcface, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, arcface, device, arcface)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Log training progress
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%\n")
            f.write(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%\n")
            f.write(f"Epoch time: {epoch_time:.2f} seconds\n")
            f.write(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}\n\n")
        
        # Print CUDA memory info
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        
        # Save checkpoint if it's the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
            }
            if arcface:
                checkpoint['arcface_state_dict'] = arcface.state_dict()
            
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {best_val_acc*100:.2f}%")
        
        # Save the latest model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
        }
        if arcface:
            checkpoint['arcface_state_dict'] = arcface.state_dict()
        
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pth'))
        
        # Clear some GPU memory at the end of each epoch
        gc.collect()
        torch.cuda.empty_cache()
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal training time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    
    with open(log_file, 'a') as f:
        f.write(f"\nTotal training time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}\n")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    print(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    main()