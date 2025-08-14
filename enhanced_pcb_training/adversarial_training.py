#!/usr/bin/env python3
"""
Enhanced Adversarial Training for PCB Defect Detection
======================================================

Implements state-of-the-art adversarial training techniques for improved 
robustness and generalization in PCB defect detection systems.

Key Features:
- FGSM and PGD adversarial attack generation
- Clean + Adversarial training (Madry et al.)
- Trades loss for robustness-accuracy trade-off
- Domain-specific adversarial examples
- Adaptive adversarial training schedules

Author: AI Research Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class AdversarialConfig:
    """Configuration for adversarial training"""
    # Attack parameters
    epsilon: float = 8/255  # L∞ perturbation bound
    alpha: float = 2/255    # Step size for PGD
    num_steps: int = 10     # Number of PGD steps
    random_start: bool = True
    
    # Training parameters
    adversarial_ratio: float = 0.5  # Ratio of adversarial examples
    clean_ratio: float = 0.5       # Ratio of clean examples
    trades_beta: float = 6.0       # TRADES regularization parameter
    
    # Schedule parameters
    warmup_epochs: int = 5         # Epochs before adversarial training
    epsilon_schedule: str = "fixed"  # "fixed", "linear", "cosine"
    
    # PCB-specific parameters
    preserve_structure: bool = True  # Preserve PCB structure in attacks
    component_aware: bool = True     # Focus attacks on components
    defect_preservation: float = 0.8 # Preserve defect characteristics

class PCBAdversarialAttacker:
    """PCB-specific adversarial attack generator"""
    
    def __init__(self, config: AdversarialConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.alpha = config.alpha
        self.num_steps = config.num_steps
        
    def fgsm_attack(self, model: nn.Module, x: torch.Tensor, 
                   y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            model: Target model
            x: Input images [batch_size, channels, height, width]
            y: True labels
            targeted: Whether to perform targeted attack
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(x_adv)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Compute loss
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        grad_sign = x_adv.grad.sign()
        
        if targeted:
            x_adv = x_adv - self.epsilon * grad_sign
        else:
            x_adv = x_adv + self.epsilon * grad_sign
        
        # Enhancement: Adaptive adversarial signals based on domain characteristics
        
        # Apply PCB-specific constraints
        if self.config.preserve_structure:
            x_adv = self._preserve_pcb_structure(x, x_adv)
        
        return torch.clamp(x_adv, 0, 1).detach()
    
    def pgd_attack(self, model: nn.Module, x: torch.Tensor, 
                   y: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack
        
        Args:
            model: Target model
            x: Input images
            y: True labels
            targeted: Whether to perform targeted attack
            
        Returns:
            Adversarial examples
        """
        x_adv = x.clone().detach()
        
        # Random initialization
        if self.config.random_start:
            noise = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv + noise, 0, 1)
        
        # PGD iterations
        for step in range(self.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = model(x_adv)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Compute loss
            loss = F.cross_entropy(outputs, y)
            
            # Enhanced: Additional regularization term for perturbation diversity
            total_loss = loss
            if self.config.trades_beta:
                kl_div = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(model(x), dim=1), reduction='batchmean')
                total_loss = loss + self.config.trades_beta * kl_div
            
            # Backward pass
            model.zero_grad()
            total_loss.backward()
            
            # Update adversarial example
            grad = x_adv.grad.data
            
            if targeted:
                x_adv = x_adv - self.alpha * grad.sign()
            else:
                x_adv = x_adv + self.alpha * grad.sign()
            
            # Project to epsilon ball
            eta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + eta, 0, 1).detach()
        
        # Apply PCB-specific constraints
        if self.config.preserve_structure:
            x_adv = self._preserve_pcb_structure(x, x_adv)
        
        return x_adv
    
    def _preserve_pcb_structure(self, x_clean: torch.Tensor, 
                               x_adv: torch.Tensor) -> torch.Tensor:
        """
        Preserve important PCB structural elements during adversarial attacks
        
        Args:
            x_clean: Original clean images
            x_adv: Adversarial images
            
        Returns:
            Structure-preserved adversarial images
        """
        # Detect high-frequency components (likely circuit traces/components)
        # Using simple edge detection as proxy for important structures
        
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(x_clean.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(x_clean.device)
        
        # Convert to grayscale for edge detection
        gray_clean = 0.299 * x_clean[:, 0:1] + 0.587 * x_clean[:, 1:2] + 0.114 * x_clean[:, 2:3]
        
        # Compute edges
        edges_x = F.conv2d(gray_clean, sobel_x, padding=1)
        edges_y = F.conv2d(gray_clean, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2)
        
        # Create structure mask (preserve high-edge areas)
        structure_mask = (edges > edges.mean() + edges.std()).float()
        structure_mask = structure_mask.repeat(1, 3, 1, 1)  # Expand to RGB
        
        # Blend adversarial and clean images in structural regions
        preservation_strength = self.config.defect_preservation
        x_preserved = (1 - structure_mask * preservation_strength) * x_adv + \
                     (structure_mask * preservation_strength) * x_clean
        
        return x_preserved
    
    def component_aware_attack(self, model: nn.Module, x: torch.Tensor, 
                              y: torch.Tensor, component_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Component-aware adversarial attack focusing on PCB components
        
        Args:
            model: Target model
            x: Input images
            y: True labels
            component_masks: Binary masks indicating component locations
            
        Returns:
            Component-focused adversarial examples
        """
        if component_masks is None:
            # Generate simple component masks using intensity thresholding
            component_masks = self._generate_component_masks(x)
        
        x_adv = x.clone().detach()
        
        # Random initialization in component regions
        if self.config.random_start:
            noise = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            noise = noise * component_masks  # Restrict noise to component areas
            x_adv = torch.clamp(x_adv + noise, 0, 1)
        
        # PGD with component focus
        for step in range(self.num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            outputs = model(x_adv)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Compute loss
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update adversarial example (focused on components)
            grad = x_adv.grad.data
            grad = grad * component_masks  # Focus gradients on components
            
            x_adv = x_adv + self.alpha * grad.sign()
            
            # Project to epsilon ball
            eta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            eta = eta * component_masks  # Restrict perturbations to components
            x_adv = torch.clamp(x + eta, 0, 1).detach()
        
        return x_adv
    
    def _generate_component_masks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate component masks using simple heuristics
        
        Args:
            x: Input images
            
        Returns:
            Binary masks indicating component locations
        """
        # Convert to HSV for better component detection
        # This is a simplified implementation
        
        # Use intensity and color variation to detect components
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        
        # Components are typically darker or lighter than PCB substrate
        substrate_intensity = gray.median()
        component_mask = torch.abs(gray - substrate_intensity) > 0.1
        
        # Expand to all channels
        component_mask = component_mask.unsqueeze(1).repeat(1, 3, 1, 1).float()
        
        return component_mask

class TRADESLoss(nn.Module):
    """
    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization) Loss
    
    Balances natural accuracy and adversarial robustness
    """
    
    def __init__(self, beta: float = 6.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, logits_natural: torch.Tensor, logits_adv: torch.Tensor, 
                y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute TRADES loss
        
        Args:
            logits_natural: Logits from clean examples
            logits_adv: Logits from adversarial examples
            y: True labels
            
        Returns:
            Total loss and loss components
        """
        # Natural loss (cross-entropy on clean examples)
        loss_natural = F.cross_entropy(logits_natural, y)
        
        # Robustness loss (KL divergence between clean and adversarial predictions)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_natural, dim=1),
            reduction='batchmean'
        )
        
        # Total TRADES loss
        loss_total = loss_natural + self.beta * loss_robust
        
        loss_dict = {
            'total': loss_total.item(),
            'natural': loss_natural.item(),
            'robust': loss_robust.item()
        }
        
        return loss_total, loss_dict

class AdversarialTrainer:
    """
    Comprehensive adversarial training framework for PCB defect detection
    """
    
    def __init__(self, model: nn.Module, config: AdversarialConfig, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.attacker = PCBAdversarialAttacker(config)
        self.trades_loss = TRADESLoss(config.trades_beta)
        
        # Training statistics
        self.training_stats = {
            'epoch': [],
            'clean_acc': [],
            'adv_acc': [],
            'natural_loss': [],
            'robust_loss': []
        }
        
        logger.info(f"Adversarial trainer initialized with epsilon={config.epsilon:.4f}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   epoch: int) -> Dict[str, float]:
        """
        Train one epoch with adversarial examples
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_natural_loss = 0.0
        total_robust_loss = 0.0
        correct_clean = 0
        correct_adv = 0
        total_samples = 0
        
        # Adjust epsilon based on schedule
        current_epsilon = self._get_scheduled_epsilon(epoch)
        self.attacker.epsilon = current_epsilon
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Determine training mode for this batch
            if epoch < self.config.warmup_epochs:
                # Warmup phase: clean training only
                use_adversarial = False
            else:
                # Mixed training with probability
                use_adversarial = random.random() < self.config.adversarial_ratio
            
            optimizer.zero_grad()
            
            if use_adversarial:
                # Generate adversarial examples
                if self.config.component_aware:
                    x_adv = self.attacker.component_aware_attack(self.model, data, target)
                else:
                    x_adv = self.attacker.pgd_attack(self.model, data, target)
                
                # Forward pass on clean and adversarial examples
                logits_clean = self.model(data)
                logits_adv = self.model(x_adv)
                
                if isinstance(logits_clean, tuple):
                    logits_clean = logits_clean[0]
                if isinstance(logits_adv, tuple):
                    logits_adv = logits_adv[0]
                
                # TRADES loss
                loss, loss_dict = self.trades_loss(logits_clean, logits_adv, target)
                
                total_natural_loss += loss_dict['natural']
                total_robust_loss += loss_dict['robust']
                
                # Compute accuracy on adversarial examples
                pred_adv = logits_adv.argmax(dim=1)
                correct_adv += pred_adv.eq(target).sum().item()
            else:
                # Clean training
                logits_clean = self.model(data)
                if isinstance(logits_clean, tuple):
                    logits_clean = logits_clean[0]
                
                loss = F.cross_entropy(logits_clean, target)
                total_natural_loss += loss.item()
            
            # Compute accuracy on clean examples
            pred_clean = logits_clean.argmax(dim=1)
            correct_clean += pred_clean.eq(target).sum().item()
            
            total_loss += loss.item()
            total_samples += target.size(0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, '
                           f'Loss: {loss.item():.4f}, '
                           f'Epsilon: {current_epsilon:.4f}')
        
        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        clean_accuracy = 100.0 * correct_clean / total_samples
        adv_accuracy = 100.0 * correct_adv / total_samples if correct_adv > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'natural_loss': total_natural_loss / len(dataloader),
            'robust_loss': total_robust_loss / len(dataloader),
            'epsilon': current_epsilon
        }
        
        # Update training statistics
        self.training_stats['epoch'].append(epoch)
        self.training_stats['clean_acc'].append(clean_accuracy)
        self.training_stats['adv_acc'].append(adv_accuracy)
        self.training_stats['natural_loss'].append(metrics['natural_loss'])
        self.training_stats['robust_loss'].append(metrics['robust_loss'])
        
        return metrics
    
    def evaluate_robustness(self, dataloader: DataLoader, 
                           attack_types: List[str] = None) -> Dict[str, float]:
        """
        Evaluate model robustness against various attacks
        
        Args:
            dataloader: Test data loader
            attack_types: List of attack types to evaluate
            
        Returns:
            Robustness metrics
        """
        if attack_types is None:
            attack_types = ['clean', 'fgsm', 'pgd', 'component_aware']
        
        self.model.eval()
        results = {}
        
        for attack_type in attack_types:
            correct = 0
            total = 0
            
            with torch.no_grad() if attack_type == 'clean' else torch.enable_grad():
                for data, target in dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if attack_type == 'clean':
                        test_data = data
                    elif attack_type == 'fgsm':
                        test_data = self.attacker.fgsm_attack(self.model, data, target)
                    elif attack_type == 'pgd':
                        test_data = self.attacker.pgd_attack(self.model, data, target)
                    elif attack_type == 'component_aware':
                        test_data = self.attacker.component_aware_attack(self.model, data, target)
                    else:
                        continue
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(test_data)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        pred = outputs.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)
            
            accuracy = 100.0 * correct / total if total > 0 else 0.0
            results[f'{attack_type}_accuracy'] = accuracy
            
            logger.info(f'{attack_type.upper()} Accuracy: {accuracy:.2f}%')
        
        return results
    
    def _get_scheduled_epsilon(self, epoch: int) -> float:
        """
        Get epsilon value based on schedule
        
        Args:
            epoch: Current epoch
            
        Returns:
            Current epsilon value
        """
        if self.config.epsilon_schedule == 'fixed':
            return self.config.epsilon
        elif self.config.epsilon_schedule == 'linear':
            # Linear increase from 0 to epsilon over warmup epochs
            if epoch < self.config.warmup_epochs:
                return self.config.epsilon * epoch / self.config.warmup_epochs
            else:
                return self.config.epsilon
        elif self.config.epsilon_schedule == 'cosine':
            # Cosine schedule
            if epoch < self.config.warmup_epochs:
                return self.config.epsilon * 0.5 * (1 + np.cos(np.pi * epoch / self.config.warmup_epochs))
            else:
                return self.config.epsilon
        else:
            return self.config.epsilon
    
    def save_training_stats(self, filepath: str):
        """
        Save training statistics
        
        Args:
            filepath: Path to save statistics
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        logger.info(f"Training statistics saved to {filepath}")

def create_adversarial_trainer(model: nn.Module, 
                              epsilon: float = 8/255,
                              num_steps: int = 10,
                              trades_beta: float = 6.0,
                              **kwargs) -> AdversarialTrainer:
    """
    Factory function to create adversarial trainer
    
    Args:
        model: Model to train
        epsilon: L∞ perturbation bound
        num_steps: Number of PGD steps
        trades_beta: TRADES regularization parameter
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured adversarial trainer
    """
    config = AdversarialConfig(
        epsilon=epsilon,
        num_steps=num_steps,
        trades_beta=trades_beta,
        **kwargs
    )
    
    trainer = AdversarialTrainer(model, config)
    
    logger.info(f"Created adversarial trainer with epsilon={epsilon:.4f}, "
               f"steps={num_steps}, beta={trades_beta}")
    
    return trainer

if __name__ == "__main__":
    # Demonstration of adversarial training
    print("🛡️ Enhanced Adversarial Training for PCB Defect Detection")
    print("=" * 60)
    
    # Example configuration
    config = AdversarialConfig(
        epsilon=8/255,
        num_steps=10,
        trades_beta=6.0,
        preserve_structure=True,
        component_aware=True
    )
    
    print(f"✅ Configuration created")
    print(f"📊 Epsilon: {config.epsilon:.4f}")
    print(f"🎯 PGD steps: {config.num_steps}")
    print(f"🔧 TRADES beta: {config.trades_beta}")
    print(f"🏗️ Structure preservation: {config.preserve_structure}")
    print(f"🔍 Component-aware attacks: {config.component_aware}")
    
    print(f"\n✨ Adversarial training ready for deployment!")
    print(f"🎯 Expected benefits: Improved robustness and generalization")
    print(f"📈 Recommended for: Industrial PCB inspection systems")
