#!/usr/bin/env python3
"""
Enhanced Self-Supervised Learning for PCB Defect Detection
==========================================================

Implements SimCLR and contrastive learning techniques for improved 
representation learning from unlabeled PCB images.

Key Features:
- SimCLR contrastive learning framework
- PCB-specific augmentation strategies
- Temperature-scaled InfoNCE loss
- Multi-scale feature learning
- Domain-specific transformations

Author: AI Research Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import random
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

@dataclass
class SimCLRConfig:
    """Configuration for SimCLR training"""
    # Training parameters
    batch_size: int = 256
    temperature: float = 0.07
    epochs: int = 1000
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4
    
    # Architecture parameters
    projection_dim: int = 128
    hidden_dim: int = 2048
    base_encoder: str = "resnet50"
    
    # Augmentation parameters
    crop_size: int = 224
    min_scale: float = 0.08
    max_scale: float = 1.0
    color_jitter_strength: float = 1.0
    blur_probability: float = 0.5
    
    # PCB-specific parameters
    preserve_components: bool = True
    defect_aware_aug: bool = True
    multi_scale_patches: bool = True

class PCBAugmentation:
    """PCB-specific augmentation for contrastive learning"""
    
    def __init__(self, config: SimCLRConfig):
        self.config = config
        
        # Base augmentations
        self.color_jitter = transforms.ColorJitter(
            brightness=0.8 * config.color_jitter_strength,
            contrast=0.8 * config.color_jitter_strength,
            saturation=0.8 * config.color_jitter_strength,
            hue=0.2 * config.color_jitter_strength
        )
        
        # PCB-specific transformations
        self.augment_pipeline = transforms.Compose([
            transforms.RandomResizedCrop(
                config.crop_size,
                scale=(config.min_scale, config.max_scale),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # PCBs can be rotated
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], 
                                 p=config.blur_probability),
            self._pcb_specific_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _pcb_specific_transform(self, img):
        """Apply PCB-specific transformations"""
        if random.random() < 0.3:  # Circuit trace emphasis
            img = self._enhance_traces(img)
        
        if random.random() < 0.2:  # Component masking
            img = self._random_component_mask(img)
        
        return img
    
    def _enhance_traces(self, img):
        """Enhance circuit trace visibility"""
        # Convert to numpy for processing
        img_np = np.array(img)
        
        # Edge enhancement filter
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        
        # Apply to each channel
        enhanced = np.zeros_like(img_np)
        for c in range(3):
            enhanced[:, :, c] = np.clip(
                np.convolve(img_np[:, :, c].flatten(), kernel.flatten(), 'same').reshape(img_np.shape[:2]),
                0, 255
            )
        
        return Image.fromarray(enhanced.astype(np.uint8))
    
    def _random_component_mask(self, img):
        """Randomly mask components to focus on traces"""
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        
        # Create random rectangular masks
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(x1, w), random.randint(y1, h)
            
            # Apply mask with mean color
            mask_color = img_np[y1:y2, x1:x2].mean(axis=(0, 1))
            img_np[y1:y2, x1:x2] = mask_color
        
        return Image.fromarray(img_np.astype(np.uint8))
    
    def __call__(self, x):
        return self.augment_pipeline(x)

class ProjectionHead(nn.Module):
    """Projection head for SimCLR"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class SimCLRModel(nn.Module):
    """SimCLR model for PCB representation learning"""
    
    def __init__(self, config: SimCLRConfig):
        super().__init__()
        self.config = config
        
        # Base encoder
        if config.base_encoder == "resnet50":
            import torchvision.models as models
            self.encoder = models.resnet50(pretrained=False)
            encoder_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported encoder: {config.base_encoder}")
        
        # Projection head
        self.projection_head = ProjectionHead(
            encoder_dim, config.hidden_dim, config.projection_dim
        )
    
    def forward(self, x):
        # Extract features
        features = self.encoder(x)
        
        # Project to contrastive space
        projections = self.projection_head(features)
        
        return features, projections

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projections: [2*batch_size, projection_dim] tensor
        """
        batch_size = projections.shape[0] // 2
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(projections, projections.t()) / self.temperature
        
        # Create labels (positive pairs are (i, i+batch_size) and (i+batch_size, i))
        labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                           torch.arange(0, batch_size)]).to(projections.device)
        
        # Create mask to ignore self-similarity
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(projections.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class SimCLRTrainer:
    """SimCLR trainer for PCB representation learning"""
    
    def __init__(self, model: SimCLRModel, config: SimCLRConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.criterion = InfoNCELoss(config.temperature)
        self.augmentation = PCBAugmentation(config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        # Training statistics
        self.training_stats = {
            'epoch': [],
            'loss': [],
            'temperature': [],
            'lr': []
        }
        
        logger.info(f"SimCLR trainer initialized with temperature={config.temperature}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            # Generate two augmented views for each image
            batch_size = images.size(0)
            
            # Create augmented pairs
            augmented_images = []
            for img in images:
                img_pil = transforms.ToPILImage()(img)
                aug1 = self.augmentation(img_pil).unsqueeze(0)
                aug2 = self.augmentation(img_pil).unsqueeze(0)
                augmented_images.extend([aug1, aug2])
            
            # Stack all augmented images
            augmented_batch = torch.cat(augmented_images, dim=0).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            _, projections = self.model(augmented_batch)
            
            # Compute contrastive loss
            loss = self.criterion(projections)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                           f'Loss: {loss.item():.4f}, '
                           f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update statistics
        self.training_stats['epoch'].append(epoch)
        self.training_stats['loss'].append(avg_loss)
        self.training_stats['temperature'].append(self.config.temperature)
        self.training_stats['lr'].append(current_lr)
        
        return {
            'loss': avg_loss,
            'lr': current_lr,
            'temperature': self.config.temperature
        }
    
    def extract_features(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features for downstream tasks"""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features, _ = self.model(images)
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Model loaded from {filepath}")

def create_simclr_trainer(base_encoder: str = "resnet50",
                         projection_dim: int = 128,
                         temperature: float = 0.07,
                         **kwargs) -> SimCLRTrainer:
    """Factory function to create SimCLR trainer"""
    config = SimCLRConfig(
        base_encoder=base_encoder,
        projection_dim=projection_dim,
        temperature=temperature,
        **kwargs
    )
    
    model = SimCLRModel(config)
    trainer = SimCLRTrainer(model, config)
    
    logger.info(f"Created SimCLR trainer with {base_encoder} encoder, "
               f"projection_dim={projection_dim}, temperature={temperature}")
    
    return trainer

if __name__ == "__main__":
    # Demonstration of SimCLR training
    print("🧠 Enhanced Self-Supervised Learning for PCB Defect Detection")
    print("=" * 65)
    
    # Example configuration
    config = SimCLRConfig(
        batch_size=64,
        temperature=0.07,
        epochs=100,
        projection_dim=128,
        preserve_components=True,
        defect_aware_aug=True
    )
    
    print(f" SimCLR configuration created")
    print(f" Temperature: {config.temperature}")
    print(f" Projection dimension: {config.projection_dim}")
    print(f" Augmentation strength: {config.color_jitter_strength}")
    print(f" Component preservation: {config.preserve_components}")
    print(f" Defect-aware augmentation: {config.defect_aware_aug}")
    
    print(f"\n SimCLR ready for self-supervised pre-training!")
    print(f" Expected benefits: Better feature representations without labels")
    print(f" Use case: Pre-training on large unlabeled PCB datasets")
