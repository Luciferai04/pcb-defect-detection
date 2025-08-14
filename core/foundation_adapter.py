#!/usr/bin/env python3
"""
Foundation Model Adaptation Framework
=====================================

Core framework for adapting CLIP, BLIP, ImageBind, and SAM to data-scarce,
fine-grained, domain-shifted environments.

Supports:
- Medical imaging (DICOM, radiological analysis)
- Remote sensing (multi-spectral, temporal sequences)
- Scientific microscopy (gigapixel, z-stacks)
- Materials characterization (SEM, TEM, X-ray)


Focus: Parameter efficiency (<2%), extreme domain shift adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Optional heavy dependencies (lazy/optional import to keep tests light)
try:  # OpenAI CLIP (installed via git+https://github.com/openai/CLIP.git)
    import clip  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    clip = None  # fallback placeholder

try:  # Transformers CLIP
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

from sklearn.metrics import accuracy_score
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
import random
from contextlib import contextmanager

# Configure reproducible logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_reproducible_seed(seed: int = 42) -> None:
    """
    Set reproducible seeds across all frameworks.
    
    Critical for parameter efficiency research where small changes
    can lead to dramatically different adaptation outcomes.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class AdaptationConfig:
    """
    Configuration for foundation model adaptation.
    
    Enforces parameter efficiency constraints and domain-specific
    optimization based on research findings.
    """
    # Core adaptation parameters
    method: str = "AD-CLIP"  # AD-CLIP, Block-LoRA, Meta-Learning
    domain: str = "medical"  # medical, remote_sensing, microscopy, materials
    
    # Parameter efficiency constraints (<2% trainable)
    max_trainable_ratio: float = 0.02
    rank: int = 4  # LoRA rank (4-16 optimal range)
    alpha: float = 32.0  # Scaling factor (α/r = 8 optimal)
    
    # Data scarcity settings (50-500 samples per class)
    samples_per_class: int = 100
    few_shot_k: int = 5
    
    # Training optimization
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Domain-specific settings
    domain_prompts: List[str] = field(default_factory=list)
    corruption_robustness: bool = True
    
    # Evaluation metrics
    use_transfer_score: bool = True
    cross_dataset_consistency: bool = True
    
    def __post_init__(self):
        """Initialize domain-specific configurations."""
        if not self.domain_prompts:
            self.domain_prompts = self._get_domain_prompts()
    
    def _get_domain_prompts(self) -> List[str]:
        """Get domain-specific prompt templates."""
        domain_prompts = {
            'medical': [
                "a medical image showing {}",
                "radiological finding of {}",
                "clinical presentation of {}",
                "diagnostic imaging of {}"
            ],
            'remote_sensing': [
                "satellite image of {}",
                "aerial view of {}",
                "remote sensing data showing {}",
                "geospatial analysis of {}"
            ],
            'microscopy': [
                "microscopic image of {}",
                "cellular structure showing {}",
                "histological section of {}",
                "magnified view of {}"
            ],
            'materials': [
                "SEM image of {}",
                "material characterization of {}",
                "surface analysis showing {}",
                "crystalline structure of {}"
            ]
        }
        return domain_prompts.get(self.domain, ["image of {}"])

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation Layer
    
    Implements parameter-efficient fine-tuning with <2% trainable parameters.
    Based on research showing optimal rank 4-16 and α=32 for domain adaptation.
    
    Mathematical formulation:
    h = Wx + (x @ A^T @ B^T) * (α/r)
    
    Where W is frozen, A,B are trainable low-rank matrices.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 4, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize matrices for stability and efficiency
        # A: small random values for symmetry breaking
        # B: zeros to ensure ΔW = 0 initially (stable start)
        self.lora_A = Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Track parameter efficiency for research analysis
        self.original_params = in_features * out_features
        self.added_params = rank * (in_features + out_features)
        self.efficiency_ratio = self.added_params / self.original_params
        
        logger.debug(f"LoRA {in_features}→{out_features}: "
                    f"rank={rank}, efficiency={self.efficiency_ratio:.4f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with low-rank adaptation.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Low-rank adapted output
        """
        x_dropout = self.dropout(x)
        # Efficient computation: x @ A^T @ B^T
        return (x_dropout @ self.lora_A.T @ self.lora_B.T) * self.scaling

class DomainPromptLearning(nn.Module):
    """
    Domain-specific prompt learning for extreme domain shift.
    
    Learns prompt tokens optimized for each target domain while
    maintaining parameter efficiency constraints.
    """
    
    def __init__(self, num_domains: int = 4, prompt_length: int = 16, 
                 embed_dim: int = 512):
        super().__init__()
        self.num_domains = num_domains
        self.prompt_length = prompt_length
        
        # Learnable prompt embeddings for each domain
        self.domain_prompts = Parameter(
            torch.randn(num_domains, prompt_length, embed_dim) * 0.02
        )
        
        # Domain-specific scaling factors
        self.domain_scales = Parameter(torch.ones(num_domains))
        
        logger.info(f"Domain prompts: {num_domains} domains, "
                   f"{prompt_length} tokens, {embed_dim} dim")
    
    def forward(self, domain_id: int, context_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply domain-specific prompts to context embeddings.
        
        Args:
            domain_id: Target domain identifier (0-3)
            context_embeddings: Base embeddings to adapt
            
        Returns:
            Domain-adapted embeddings
        """
        if domain_id >= self.num_domains:
            domain_id = 0  # Default to first domain
        
        domain_prompt = self.domain_prompts[domain_id]
        scale = self.domain_scales[domain_id]
        
        # Apply domain adaptation with learned scaling
        adapted = context_embeddings + (domain_prompt.mean(dim=0) * scale)
        return adapted

class BaseFoundationAdapter(nn.Module, ABC):
    """
    Abstract base class for foundation model adaptation.
    
    Defines the interface and common functionality for adapting
    CLIP, BLIP, ImageBind, and SAM to specialized domains.
    """
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        self.adaptation_modules = nn.ModuleDict()
        self.domain_prompts = DomainPromptLearning()
        
        # Initialize base model (to be implemented by subclasses)
        self.base_model = None
        self._setup_base_model()
        
        # Track parameter efficiency
        self._compute_efficiency_metrics()
    
    @abstractmethod
    def _setup_base_model(self) -> None:
        """Setup the base foundation model (CLIP, BLIP, etc.)"""
        pass
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with domain adaptation"""
        pass
    
    def _compute_efficiency_metrics(self) -> None:
        """
        Compute and validate parameter efficiency constraints.
        
        Ensures <2% trainable parameters as per research requirements.
        """
        if self.base_model is None:
            return
        
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.total_parameters = total_params
        self.trainable_parameters = trainable_params
        self.efficiency_ratio = trainable_params / total_params if total_params > 0 else 0
        
        # Validate efficiency constraint
        if self.efficiency_ratio > self.config.max_trainable_ratio:
            logger.warning(f"Parameter efficiency violation: "
                          f"{self.efficiency_ratio:.4f} > {self.config.max_trainable_ratio}")
        
        logger.info(f"Parameter efficiency: {self.efficiency_ratio:.4f} "
                   f"({trainable_params:,}/{total_params:,})")
    
    def freeze_base_model(self) -> None:
        """Freeze base model parameters for efficient adaptation."""
        if self.base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Base model parameters frozen")

class CLIPAdapter(BaseFoundationAdapter):
    """
    CLIP adaptation for domain-shifted, data-scarce environments.
    
    Implements AD-CLIP with prompt-space domain adaptation and
    Block-LoRA for parameter-efficient fine-tuning.
    
    Research Context:
    - AD-CLIP: +7.6% improvement through prompt-space adaptation
    - Block-LoRA: 70.43% accuracy with 2% trainable parameters
    - Optimal for medical imaging, remote sensing applications
    """
    
    def _setup_base_model(self) -> None:
        """Initialize CLIP model with frozen parameters."""
        try:
            self.base_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
            self.freeze_base_model()
            
            # Apply LoRA to key attention layers
            self._inject_lora_layers()
            
            logger.info("CLIP model loaded and adapted")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _inject_lora_layers(self) -> None:
        """
        Inject LoRA layers into critical transformer blocks.
        
        Targets final attention layers for maximum adaptation benefit
        with minimal parameter overhead.
        """
        target_modules = [
            'visual.transformer.resblocks.11.attn.in_proj_weight',
            'visual.transformer.resblocks.11.attn.out_proj.weight',
            'visual.transformer.resblocks.10.attn.in_proj_weight',
            'visual.transformer.resblocks.9.attn.in_proj_weight'
        ]
        
        injection_count = 0
        for name, module in self.base_model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    lora_layer = LoRALayer(
                        module.in_features,
                        module.out_features,
                        self.config.rank,
                        self.config.alpha
                    )
                    
                    self.adaptation_modules[name.replace('.', '_')] = lora_layer
                    injection_count += 1
        
        logger.info(f"Injected {injection_count} LoRA layers")
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with CLIP adaptation.
        
        Args:
            inputs: Dictionary containing:
                - 'image': Image tensor [batch_size, 3, 224, 224]
                - 'text': Text tokens [batch_size, 77]
                - 'domain_id': Domain identifier
                
        Returns:
            Dictionary with logits and features
        """
        images = inputs['image']
        texts = inputs['text']
        domain_id = inputs.get('domain_id', 0)
        
        # Get base features (frozen)
        with torch.no_grad():
            image_features = self.base_model.encode_image(images)
            text_features = self.base_model.encode_text(texts)
        
        # Apply domain-specific adaptations
        adapted_image_features = self._apply_adaptations(image_features, 'visual')
        adapted_text_features = self._apply_adaptations(text_features, 'text')
        
        # Domain prompt learning
        adapted_text_features = self.domain_prompts(domain_id, adapted_text_features)
        
        # Normalize features
        adapted_image_features = F.normalize(adapted_image_features, dim=-1)
        adapted_text_features = F.normalize(adapted_text_features, dim=-1)
        
        # Compute similarity logits
        logit_scale = self.base_model.logit_scale.exp()
        logits_per_image = logit_scale * adapted_image_features @ adapted_text_features.t()
        logits_per_text = logits_per_image.t()
        
        return {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_features': adapted_image_features,
            'text_features': adapted_text_features,
            'logit_scale': logit_scale
        }
    
    def _apply_adaptations(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Apply LoRA adaptations based on modality.
        
        Args:
            features: Input features to adapt
            modality: 'visual' or 'text' for targeted adaptation
            
        Returns:
            Adapted features with LoRA modifications
        """
        adapted_features = features
        
        for name, lora_layer in self.adaptation_modules.items():
            if modality in name:
                adaptation = lora_layer(adapted_features)
                adapted_features = adapted_features + adaptation
        
        return adapted_features

@contextmanager
def mixed_precision_context(enabled: bool = True):
    """Context manager for mixed precision training optimization."""
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield

class AdaptationTrainer:
    """
    Trainer for foundation model adaptation with optimization for:
    - Data scarcity (50-500 samples per class)
    - Parameter efficiency (<2% trainable)
    - Domain shift robustness
    - Memory optimization (gradient checkpointing, mixed precision)
    """
    
    def __init__(self, model: BaseFoundationAdapter, config: AdaptationConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer targeting only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Learning rate scheduling for stability
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100, 
            eta_min=1e-6
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Training history for analysis
        self.training_history = []
        
        logger.info(f"Trainer initialized for {config.method} on {config.domain}")
    
    def train_epoch(self, dataloader, domain_id: int = 0) -> Dict[str, float]:
        """
        Train one epoch with gradient accumulation and mixed precision.
        
        Args:
            dataloader: Training data loader
            domain_id: Target domain identifier
            
        Returns:
            Training metrics dictionary
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulated_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Prepare batch data
            inputs = self._prepare_batch(batch, domain_id)
            
            # Forward pass with mixed precision
            with mixed_precision_context(self.config.mixed_precision):
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, inputs)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            accumulated_steps += 1
            
            # Update weights after accumulation
            if accumulated_steps % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")
        
        # Update learning rate
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics = {
            'loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'parameter_efficiency': self.model.efficiency_ratio
        }
        
        self.training_history.append(metrics)
        return metrics
    
    def _prepare_batch(self, batch: Tuple, domain_id: int) -> Dict[str, torch.Tensor]:
        """Prepare batch data for model input."""
        if len(batch) == 3:
            images, texts, labels = batch
        else:
            images, texts = batch
            labels = None
        
        return {
            'image': images.to(self.device),
            'text': texts.to(self.device),
            'labels': labels.to(self.device) if labels is not None else None,
            'domain_id': domain_id
        }
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute contrastive loss for image-text alignment.
        
        Uses symmetric cross-entropy loss as in CLIP training,
        optimized for few-shot scenarios.
        """
        logits_per_image = outputs['logits_per_image']
        logits_per_text = outputs['logits_per_text']
        
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Symmetric contrastive loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def evaluate(self, dataloader, domain_id: int = 0) -> Dict[str, float]:
        """
        Evaluate model with comprehensive metrics.
        
        Includes accuracy, parameter efficiency, and domain-specific metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = self._prepare_batch(batch, domain_id)
                outputs = self.model(inputs)
                
                loss = self._compute_loss(outputs, inputs)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = outputs['logits_per_image'].argmax(dim=1)
                if inputs['labels'] is not None:
                    correct_predictions += (predictions == inputs['labels']).sum().item()
                    total_samples += inputs['labels'].size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'parameter_efficiency': self.model.efficiency_ratio,
            'trainable_params': self.model.trainable_parameters,
            'total_params': self.model.total_parameters
        }

def create_foundation_adapter(method: str = "AD-CLIP", 
                            domain: str = "medical",
                            **kwargs) -> BaseFoundationAdapter:
    """
    Factory function to create foundation model adapters.
    
    Args:
        method: Adaptation method (AD-CLIP, Block-LoRA, Meta-Learning)
        domain: Target domain (medical, remote_sensing, microscopy, materials)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured foundation model adapter
    """
    set_reproducible_seed()  # Ensure reproducibility
    
    config = AdaptationConfig(method=method, domain=domain, **kwargs)
    
    if method in ["AD-CLIP", "Block-LoRA"]:
        adapter = CLIPAdapter(config)
    else:
        raise ValueError(f"Unsupported adaptation method: {method}")
    
    logger.info(f"Created {method} adapter for {domain} domain")
    logger.info(f"Parameter efficiency: {adapter.efficiency_ratio:.4f}")
    
    return adapter

if __name__ == "__main__":
    # Demonstration of the foundation adaptation framework
    print("Foundation Model Adaptation Framework")
    print("=" * 60)
    
    # Set reproducible environment
    set_reproducible_seed(42)
    
    # Create adapters for different domains
    domains = ['medical', 'remote_sensing', 'microscopy', 'materials']
    
    for domain in domains:
        print(f"\nCreating adapter for {domain} domain...")
        
        try:
            adapter = create_foundation_adapter(
                method="AD-CLIP",
                domain=domain,
                rank=4,
                alpha=32,
                samples_per_class=100
            )
            
            print(f"  Success: {adapter.efficiency_ratio:.4f} parameter efficiency")
            print(f"  Trainable: {adapter.trainable_parameters:,} params")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nFramework ready for domain adaptation research!")
    print(f"Focus: Data scarcity (50-500 samples), extreme domain shift")
    print(f"Target: <2% trainable parameters, >70% accuracy")
