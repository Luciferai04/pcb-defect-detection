"""
Test-Time Adaptation for Robust Domain Generalization
Implements comprehensive test-time adaptation strategies for PCB defect detection with advanced domain shift handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import copy
from collections import deque

@dataclass
class TTAConfig:
    """Configuration for Test-Time Adaptation"""
    adaptation_steps: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9
    temperature: float = 1.0
    confidence_threshold: float = 0.9
    memory_size: int = 64
    augmentation_strength: float = 0.1
    bn_adaptation: bool = True
    entropy_weight: float = 1.0
    consistency_weight: float = 1.0
    
class BatchNormAdaptation:
    """Batch normalization statistics adaptation"""
    
    def __init__(self, model: nn.Module, momentum: float = 0.1):
        self.model = model
        self.momentum = momentum
        self.original_bn_stats = {}
        self._save_original_stats()
    
    def _save_original_stats(self):
        """Save original batch norm statistics"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.original_bn_stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'momentum': module.momentum
                }
    
    def adapt_bn_stats(self, x: torch.Tensor):
        """Adapt batch normalization statistics"""
        self.model.train()  # Enable BN adaptation
        
        # Forward pass to update BN statistics
        with torch.no_grad():
            _ = self.model(x)
        
        # Update momentum for faster adaptation
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.momentum = self.momentum
    
    def reset_bn_stats(self):
        """Reset batch norm statistics to original values"""
        for name, module in self.model.named_modules():
            if name in self.original_bn_stats:
                module.running_mean.copy_(self.original_bn_stats[name]['running_mean'])
                module.running_var.copy_(self.original_bn_stats[name]['running_var'])
                module.momentum = self.original_bn_stats[name]['momentum']

class TestTimeAugmentation:
    """Test-time augmentation for consistent predictions"""
    
    def __init__(self, num_augmentations: int = 8, strength: float = 0.1):
        self.num_augmentations = num_augmentations
        self.strength = strength
    
    def augment_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply test-time augmentations"""
        augmented_samples = []
        
        for _ in range(self.num_augmentations):
            # Add noise
            noise = torch.randn_like(x) * self.strength
            augmented = x + noise
            
            # Random rotation (for image data)
            if len(x.shape) == 4:  # Batch, Channel, Height, Width
                angle = np.random.uniform(-15, 15)  # degrees
                augmented = self._rotate_batch(augmented, angle)
            
            augmented_samples.append(augmented)
        
        return torch.stack(augmented_samples, dim=1)  # [batch, augmentations, ...]
    
    def _rotate_batch(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate batch of images"""
        # Simple rotation implementation (can be replaced with more sophisticated methods)
        return x  # Placeholder - implement actual rotation
    
    def ensemble_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Ensemble predictions from augmented samples"""
        # predictions shape: [batch, augmentations, classes]
        return torch.mean(predictions, dim=1)

class EntropyMinimization(nn.Module):
    """Advanced entropy minimization with temperature scaling"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss with temperature scaling"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy.mean()

class ConsistencyLoss(nn.Module):
    """Consistency loss for test-time adaptation"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits_weak: torch.Tensor, logits_strong: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between weak and strong augmentations"""
        probs_weak = F.softmax(logits_weak / self.temperature, dim=-1)
        log_probs_strong = F.log_softmax(logits_strong / self.temperature, dim=-1)
        
        consistency_loss = F.kl_div(log_probs_strong, probs_weak, reduction='batchmean')
        return consistency_loss

class MemoryBank:
    """Memory bank for storing reliable predictions"""
    
    def __init__(self, capacity: int = 64, feature_dim: int = 2048):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.features = deque(maxlen=capacity)
        self.labels = deque(maxlen=capacity)
        self.confidences = deque(maxlen=capacity)
    
    def update(self, features: torch.Tensor, predictions: torch.Tensor, confidences: torch.Tensor):
        """Update memory bank with new samples"""
        batch_size = features.size(0)
        
        for i in range(batch_size):
            if confidences[i] > 0.9:  # Only store high-confidence samples
                self.features.append(features[i].detach().cpu())
                self.labels.append(predictions[i].detach().cpu())
                self.confidences.append(confidences[i].detach().cpu())
    
    def get_reliable_samples(self, k: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get k most reliable samples from memory"""
        if len(self.features) < k:
            return None, None
        
        # Sort by confidence and get top k
        sorted_indices = sorted(range(len(self.confidences)), 
                               key=lambda i: self.confidences[i], reverse=True)
        top_k_indices = sorted_indices[:k]
        
        features = torch.stack([self.features[i] for i in top_k_indices])
        labels = torch.stack([self.labels[i] for i in top_k_indices])
        
        return features, labels

class TestTimeAdaptation:
    """Comprehensive Test-Time Adaptation framework"""
    
    def __init__(self, model: nn.Module, config: TTAConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize components
        self.bn_adaptation = BatchNormAdaptation(model, config.momentum)
        self.tta_augmentation = TestTimeAugmentation()
        self.entropy_loss = EntropyMinimization(config.temperature)
        self.consistency_loss = ConsistencyLoss(config.temperature)
        self.memory_bank = MemoryBank(config.memory_size)
        
        # Save original model state
        self.original_state_dict = copy.deepcopy(model.state_dict())
    
    def adapt_single_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt to a single batch of test data"""
        x = x.to(self.device)
        
        # Batch normalization adaptation
        if self.config.bn_adaptation:
            self.bn_adaptation.adapt_bn_stats(x)
        
        # Create weak and strong augmentations
        x_weak = x + torch.randn_like(x) * 0.01  # Weak augmentation
        x_strong = x + torch.randn_like(x) * self.config.augmentation_strength  # Strong augmentation
        
        # Adaptation loop
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            momentum=self.config.momentum
        )
        
        for step in range(self.config.adaptation_steps):
            optimizer.zero_grad()
            
            # Forward passes
            logits_weak = self.model(x_weak)
            logits_strong = self.model(x_strong)
            
            # Compute losses
            entropy_loss = self.entropy_loss(logits_weak)
            consistency_loss = self.consistency_loss(logits_weak, logits_strong)
            
            # Memory-based loss (if available)
            memory_loss = 0
            memory_features, memory_labels = self.memory_bank.get_reliable_samples()
            if memory_features is not None:
                memory_features = memory_features.to(self.device)
                memory_labels = memory_labels.to(self.device)
                memory_logits = self.model(memory_features)
                memory_loss = F.cross_entropy(memory_logits, memory_labels.argmax(dim=-1))
            
            # Total loss
            total_loss = (self.config.entropy_weight * entropy_loss + 
                         self.config.consistency_weight * consistency_loss + 
                         0.1 * memory_loss)
            
            total_loss.backward()
            optimizer.step()
        
        # Final prediction
        self.model.eval()
        with torch.no_grad():
            final_logits = self.model(x)
            final_probs = F.softmax(final_logits, dim=-1)
            
            # Update memory bank
            confidences = torch.max(final_probs, dim=-1)[0]
            features = x  # In practice, extract features from intermediate layers
            self.memory_bank.update(features, final_probs, confidences)
        
        return final_logits
    
    def adapt_with_tta(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt with test-time augmentation ensemble"""
        x = x.to(self.device)
        
        # Generate augmented samples
        x_augmented = self.tta_augmentation.augment_batch(x)
        batch_size, num_augs = x_augmented.shape[:2]
        
        # Reshape for batch processing
        x_flat = x_augmented.view(-1, *x_augmented.shape[2:])
        
        # Adapt on augmented samples
        adapted_logits = self.adapt_single_batch(x_flat)
        
        # Reshape back and ensemble
        adapted_logits = adapted_logits.view(batch_size, num_augs, -1)
        ensemble_logits = self.tta_augmentation.ensemble_predictions(adapted_logits)
        
        return ensemble_logits
    
    def reset_model(self):
        """Reset model to original state"""
        self.model.load_state_dict(self.original_state_dict)
        self.bn_adaptation.reset_bn_stats()
        self.memory_bank = MemoryBank(self.config.memory_size)
    
    def continual_adaptation(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Continual adaptation across multiple batches"""
        all_predictions = []
        
        for batch_idx, (x, _) in enumerate(dataloader):
            # Adapt to current batch
            adapted_logits = self.adapt_single_batch(x)
            all_predictions.append(adapted_logits)
            
            # Periodic model reset to prevent drift
            if batch_idx % 50 == 0 and batch_idx > 0:
                self.reset_model()
        
        return all_predictions
