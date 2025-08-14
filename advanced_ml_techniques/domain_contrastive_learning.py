"""
Domain-Specific Contrastive Learning
Implements advanced contrastive learning methods tailored for PCB defect detection with domain-specific features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from collections import defaultdict

@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning"""
    temperature: float = 0.07
    embedding_dim: int = 512
    projection_dim: int = 128
    num_negatives: int = 16384
    momentum: float = 0.999
    queue_size: int = 4096
    hard_negative_ratio: float = 0.3
    domain_weight: float = 1.0
    instance_weight: float = 1.0
    
class MomentumEncoder(nn.Module):
    """Momentum encoder for contrastive learning"""
    
    def __init__(self, encoder: nn.Module, momentum: float = 0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Create momentum encoder
        self.momentum_encoder = self._create_momentum_encoder()
        
        # Initialize momentum encoder with same weights
        self._init_momentum_encoder()
    
    def _create_momentum_encoder(self):
        """Create momentum encoder as a copy of the main encoder"""
        return type(self.encoder)(self.encoder.config) if hasattr(self.encoder, 'config') else None
    
    def _init_momentum_encoder(self):
        """Initialize momentum encoder with main encoder weights"""
        if self.momentum_encoder is not None:
            for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder parameters"""
        if self.momentum_encoder is not None:
            for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def forward(self, x: torch.Tensor, use_momentum: bool = False) -> torch.Tensor:
        """Forward pass through encoder"""
        if use_momentum and self.momentum_encoder is not None:
            return self.momentum_encoder(x)
        else:
            return self.encoder(x)
    
    def update_momentum(self):
        """Update momentum encoder"""
        self._momentum_update()

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=-1)

class MemoryQueue:
    """Memory queue for storing negative samples"""
    
    def __init__(self, size: int, dim: int, device: str = 'cuda'):
        self.size = size
        self.dim = dim
        self.device = device
        
        # Initialize queue
        self.queue = torch.randn(dim, size).to(device)
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = torch.zeros(1, dtype=torch.long).to(device)
    
    @torch.no_grad()
    def update(self, keys: torch.Tensor):
        """Update memory queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest entries
        if ptr + batch_size <= self.size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # Wrap around
            end_size = self.size - ptr
            self.queue[:, ptr:] = keys[:end_size].T
            self.queue[:, :batch_size - end_size] = keys[end_size:].T
        
        # Update pointer
        self.queue_ptr[0] = (ptr + batch_size) % self.size
    
    def get_negatives(self, num_negatives: Optional[int] = None) -> torch.Tensor:
        """Get negative samples from queue"""
        if num_negatives is None:
            return self.queue.clone().detach()
        else:
            indices = torch.randperm(self.size)[:num_negatives]
            return self.queue[:, indices].clone().detach()

class DomainAwareContrastiveLoss(nn.Module):
    """Domain-aware contrastive loss for PCB defect detection"""
    
    def __init__(self, config: ContrastiveConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        
    def forward(self, 
                queries: torch.Tensor, 
                keys: torch.Tensor, 
                negatives: torch.Tensor,
                domain_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute domain-aware contrastive loss
        
        Args:
            queries: Query embeddings [batch_size, dim]
            keys: Key embeddings [batch_size, dim]  
            negatives: Negative embeddings [dim, num_negatives]
            domain_labels: Domain labels for domain-specific weighting
        """
        batch_size = queries.size(0)
        
        # Positive similarities
        pos_sim = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.einsum('nc,ck->nk', [queries, negatives]) / self.temperature
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels (positive is always at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long).to(queries.device)
        
        # Basic contrastive loss
        loss = F.cross_entropy(logits, labels)
        
        # Domain-specific weighting
        if domain_labels is not None:
            domain_weights = self._compute_domain_weights(domain_labels)
            loss = loss * domain_weights.mean()
        
        return loss
    
    def _compute_domain_weights(self, domain_labels: torch.Tensor) -> torch.Tensor:
        """Compute domain-specific weights"""
        # Simple uniform weighting for now - can be made more sophisticated
        return torch.ones_like(domain_labels, dtype=torch.float32)

class HardNegativeMining:
    """Hard negative mining for improved contrastive learning"""
    
    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio
    
    def mine_hard_negatives(self, 
                           queries: torch.Tensor, 
                           negatives: torch.Tensor,
                           k: Optional[int] = None) -> torch.Tensor:
        """Mine hard negatives based on similarity scores"""
        # Compute similarities
        similarities = torch.einsum('nc,ck->nk', [queries, negatives])
        
        # Number of hard negatives to select
        if k is None:
            k = int(negatives.size(1) * self.ratio)
        
        # Select top-k most similar (hardest) negatives
        _, hard_indices = torch.topk(similarities, k, dim=1)
        
        # Gather hard negatives
        batch_size = queries.size(0)
        hard_negatives = []
        
        for i in range(batch_size):
            hard_neg = negatives[:, hard_indices[i]]  # [dim, k]
            hard_negatives.append(hard_neg.T)  # [k, dim]
        
        return torch.stack(hard_negatives)  # [batch_size, k, dim]

class MultiScaleContrastiveLoss(nn.Module):
    """Multi-scale contrastive loss for hierarchical feature learning"""
    
    def __init__(self, scales: List[int], temperature: float = 0.07):
        super().__init__()
        self.scales = scales
        self.temperature = temperature
    
    def forward(self, 
                features_list: List[torch.Tensor], 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale contrastive loss
        
        Args:
            features_list: List of features at different scales
            targets: Target labels for positive pair identification
        """
        total_loss = 0
        
        for scale_idx, features in enumerate(features_list):
            # Normalize features
            features = F.normalize(features, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(features, features.T) / self.temperature
            
            # Create positive pair mask
            pos_mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()
            pos_mask.fill_diagonal_(0)  # Remove self-similarities
            
            # Compute InfoNCE loss
            exp_sim = torch.exp(sim_matrix)
            neg_mask = 1 - pos_mask - torch.eye(targets.size(0)).to(features.device)
            
            # Positive similarities
            pos_sim = (exp_sim * pos_mask).sum(dim=1)
            
            # All similarities (positive + negative)
            all_sim = (exp_sim * (pos_mask + neg_mask)).sum(dim=1)
            
            # InfoNCE loss
            loss = -torch.log(pos_sim / (all_sim + 1e-8)).mean()
            total_loss += loss
        
        return total_loss / len(features_list)

class DomainSpecificContrastiveLearning(nn.Module):
    """Complete domain-specific contrastive learning framework"""
    
    def __init__(self, 
                 backbone: nn.Module, 
                 config: ContrastiveConfig,
                 device: str = 'cuda'):
        super().__init__()
        self.config = config
        self.device = device
        
        # Backbone encoder
        self.backbone = backbone
        
        # Feature dimensions (assuming backbone outputs features)
        feature_dim = getattr(backbone, 'feature_dim', 2048)
        
        # Projection heads
        self.projection_head = ProjectionHead(
            feature_dim, config.embedding_dim, config.projection_dim
        )
        
        # Momentum encoder
        self.momentum_encoder = MomentumEncoder(backbone, config.momentum)
        self.momentum_projection = ProjectionHead(
            feature_dim, config.embedding_dim, config.projection_dim
        )
        
        # Memory queue
        self.memory_queue = MemoryQueue(
            config.queue_size, config.projection_dim, device
        )
        
        # Loss functions
        self.contrastive_loss = DomainAwareContrastiveLoss(config)
        self.multi_scale_loss = MultiScaleContrastiveLoss([7, 14, 28])
        
        # Hard negative mining
        self.hard_negative_miner = HardNegativeMining(config.hard_negative_ratio)
        
        # Move to device
        self.to(device)
    
    def forward(self, 
                x_query: torch.Tensor, 
                x_key: torch.Tensor,
                domain_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning
        
        Args:
            x_query: Query images
            x_key: Key images (augmented versions)
            domain_labels: Domain labels for domain-specific learning
        """
        # Extract features
        q_features = self.backbone(x_query)
        
        # Projection
        q_proj = self.projection_head(q_features)
        
        # Momentum encoder for keys
        with torch.no_grad():
            k_features = self.momentum_encoder(x_key, use_momentum=True)
            k_proj = self.momentum_projection(k_features).detach()
        
        # Get negatives from memory queue
        negatives = self.memory_queue.get_negatives(self.config.num_negatives)
        
        # Hard negative mining
        hard_negatives = self.hard_negative_miner.mine_hard_negatives(q_proj, negatives)
        
        # Combine all negatives
        all_negatives = torch.cat([
            negatives.T,  # [num_negatives, dim]
            hard_negatives.view(-1, hard_negatives.size(-1))  # [batch*k, dim]
        ], dim=0).T  # [dim, total_negatives]
        
        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss(q_proj, k_proj, all_negatives, domain_labels)
        
        # Multi-scale contrastive loss (if backbone provides multi-scale features)
        multi_scale_loss = 0
        if hasattr(self.backbone, 'get_multi_scale_features'):
            q_multi_features = self.backbone.get_multi_scale_features(x_query)
            if domain_labels is not None:
                multi_scale_loss = self.multi_scale_loss(q_multi_features, domain_labels)
        
        # Update memory queue
        self.memory_queue.update(k_proj)
        
        # Update momentum encoder
        self.momentum_encoder.update_momentum()
        
        # Total loss
        total_loss = (self.config.instance_weight * contrastive_loss + 
                     self.config.domain_weight * multi_scale_loss)
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'multi_scale_loss': multi_scale_loss,
            'query_features': q_features,
            'query_projections': q_proj,
            'key_projections': k_proj
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings for inference"""
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.projection_head(features)
        return embeddings
    
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two inputs"""
        emb1 = self.get_embeddings(x1)
        emb2 = self.get_embeddings(x2)
        
        similarity = torch.cosine_similarity(emb1, emb2, dim=-1)
        return similarity

class PCBDefectContrastiveDataset:
    """Dataset wrapper for PCB defect contrastive learning"""
    
    def __init__(self, data, labels, domain_labels=None, augmentation_fn=None):
        self.data = data
        self.labels = labels
        self.domain_labels = domain_labels
        self.augmentation_fn = augmentation_fn
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Create positive pair through augmentation
        if self.augmentation_fn:
            aug_sample = self.augmentation_fn(sample)
        else:
            aug_sample = sample  # No augmentation
        
        result = {
            'query': sample,
            'key': aug_sample,
            'label': label
        }
        
        if self.domain_labels is not None:
            result['domain_label'] = self.domain_labels[idx]
        
        return result

class ContrastiveTrainer:
    """Trainer for domain-specific contrastive learning"""
    
    def __init__(self, 
                 model: DomainSpecificContrastiveLearning,
                 config: ContrastiveConfig,
                 device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_contrastive_loss = 0
        total_multi_scale_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            x_query = batch['query'].to(self.device)
            x_key = batch['key'].to(self.device)
            domain_labels = batch.get('domain_label', None)
            if domain_labels is not None:
                domain_labels = domain_labels.to(self.device)
            
            # Forward pass
            outputs = self.model(x_query, x_key, domain_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            outputs['total_loss'].backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += outputs['total_loss'].item()
            total_contrastive_loss += outputs['contrastive_loss'].item()
            total_multi_scale_loss += outputs['multi_scale_loss'].item()
        
        # Update learning rate
        self.scheduler.step()
        
        num_batches = len(dataloader)
        return {
            'total_loss': total_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches,
            'multi_scale_loss': total_multi_scale_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def evaluate_embeddings(self, dataloader) -> Dict[str, float]:
        """Evaluate embedding quality"""
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch['query'].to(self.device)
                labels = batch['label']
                
                embeddings = self.model.get_embeddings(x)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        # Concatenate all embeddings and labels
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute embedding metrics
        metrics = self._compute_embedding_metrics(all_embeddings, all_labels)
        
        return metrics
    
    def _compute_embedding_metrics(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute embedding quality metrics"""
        # Compute pairwise similarities
        similarities = torch.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
        )
        
        # Create label matrix
        label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        
        # Compute average positive and negative similarities
        pos_mask = label_matrix - torch.eye(len(labels))  # Remove diagonal
        neg_mask = 1 - label_matrix
        
        pos_similarities = similarities[pos_mask.bool()].mean().item()
        neg_similarities = similarities[neg_mask.bool()].mean().item()
        
        # Compute separation score
        separation = pos_similarities - neg_similarities
        
        return {
            'positive_similarity': pos_similarities,
            'negative_similarity': neg_similarities,
            'separation_score': separation
        }

# Configuration and usage example
def create_contrastive_config() -> ContrastiveConfig:
    """Create default contrastive learning configuration"""
    return ContrastiveConfig(
        temperature=0.07,
        embedding_dim=512,
        projection_dim=128,
        num_negatives=16384,
        momentum=0.999,
        queue_size=4096,
        hard_negative_ratio=0.3,
        domain_weight=1.0,
        instance_weight=1.0
    )

# Example usage
if __name__ == "__main__":
    # Create configuration
    config = create_contrastive_config()
    
    # Dummy backbone model
    backbone = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    )
    backbone.feature_dim = 512  # Set feature dimension
    
    # Create contrastive learning model
    model = DomainSpecificContrastiveLearning(backbone, config)
    
    # Test with dummy data
    x_query = torch.randn(8, 2048)
    x_key = torch.randn(8, 2048)
    domain_labels = torch.randint(0, 5, (8,))
    
    # Forward pass
    outputs = model(x_query, x_key, domain_labels)
    
    print("Domain-Specific Contrastive Learning Results:")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Contrastive loss: {outputs['contrastive_loss'].item():.4f}")
    print(f"Multi-scale loss: {outputs['multi_scale_loss'].item():.4f}")
    print(f"Query features shape: {outputs['query_features'].shape}")
    print(f"Query projections shape: {outputs['query_projections'].shape}")
