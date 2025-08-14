#!/usr/bin/env python3
"""
Test suite for Foundation Model Adapter
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Import the modules to test
from core.foundation_adapter import (
    LoRALayer, 
    DomainPromptLearning, 
    AdaptationConfig,
    set_reproducible_seed
)


class TestAdaptationConfig:
    """Test AdaptationConfig class"""
    
    def test_default_config(self):
        """Test default configuration initialization"""
        config = AdaptationConfig()
        assert config.method == "AD-CLIP"
        assert config.domain == "medical"
        assert config.max_trainable_ratio == 0.02
        assert config.rank == 4
        assert config.alpha == 32.0
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = AdaptationConfig(
            method="Block-LoRA",
            domain="microscopy",
            rank=8,
            alpha=16.0
        )
        assert config.method == "Block-LoRA"
        assert config.domain == "microscopy"
        assert config.rank == 8
        assert config.alpha == 16.0
        
    def test_domain_prompts_initialization(self):
        """Test domain-specific prompt initialization"""
        config = AdaptationConfig(domain="medical")
        assert len(config.domain_prompts) > 0
        assert any("medical" in prompt for prompt in config.domain_prompts)


class TestLoRALayer:
    """Test LoRA Layer implementation"""
    
    @pytest.fixture
    def lora_layer(self):
        """Create a test LoRA layer"""
        return LoRALayer(in_features=256, out_features=128, rank=4, alpha=32.0)
    
    def test_initialization(self, lora_layer):
        """Test LoRA layer initialization"""
        assert lora_layer.rank == 4
        assert lora_layer.alpha == 32.0
        assert lora_layer.scaling == 32.0 / 4
        assert lora_layer.lora_A.shape == (4, 256)
        assert lora_layer.lora_B.shape == (128, 4)
    
    def test_forward_pass(self, lora_layer):
        """Test LoRA forward pass"""
        batch_size = 8
        input_tensor = torch.randn(batch_size, 256)
        
        output = lora_layer(input_tensor)
        
        assert output.shape == (batch_size, 128)
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_parameter_efficiency(self, lora_layer):
        """Test parameter efficiency calculation"""
        original_params = 256 * 128  # in_features * out_features
        lora_params = 4 * (256 + 128)  # rank * (in_features + out_features)
        
        assert lora_layer.original_params == original_params
        assert lora_layer.added_params == lora_params
        assert lora_layer.efficiency_ratio == lora_params / original_params
        assert lora_layer.efficiency_ratio < 0.1  # Should be much less than full parameters
    
    @pytest.mark.gpu
    def test_gpu_compatibility(self, lora_layer):
        """Test LoRA layer GPU compatibility"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            lora_layer = lora_layer.to(device)
            input_tensor = torch.randn(8, 256, device=device)
            
            output = lora_layer(input_tensor)
            
            assert output.device == device
            assert output.shape == (8, 128)


class TestDomainPromptLearning:
    """Test Domain Prompt Learning module"""
    
    @pytest.fixture
    def prompt_learner(self):
        """Create a test domain prompt learner"""
        return DomainPromptLearning(num_domains=4, prompt_length=16, embed_dim=512)
    
    def test_initialization(self, prompt_learner):
        """Test domain prompt learning initialization"""
        assert prompt_learner.num_domains == 4
        assert prompt_learner.prompt_length == 16
        assert prompt_learner.domain_prompts.shape == (4, 16, 512)
        assert prompt_learner.domain_scales.shape == (4,)
    
    def test_parameter_count(self, prompt_learner):
        """Test that prompt learning adds minimal parameters"""
        total_params = sum(p.numel() for p in prompt_learner.parameters())
        expected_params = (4 * 16 * 512) + 4  # prompts + scales
        assert total_params == expected_params


class TestReproducibility:
    """Test reproducibility functions"""
    
    def test_set_reproducible_seed(self):
        """Test reproducible seed setting"""
        # Set seed and generate random numbers
        set_reproducible_seed(42)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)
        
        # Reset seed and generate again
        set_reproducible_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        set_reproducible_seed(42)
        result1 = torch.rand(10)
        
        set_reproducible_seed(123)
        result2 = torch.rand(10)
        
        assert not torch.allclose(result1, result2)


class TestIntegration:
    """Integration tests for foundation adapter components"""
    
    def test_lora_with_linear_layer(self):
        """Test LoRA integration with linear layer"""
        # Create a linear layer and LoRA adaptation
        linear = nn.Linear(256, 128)
        lora = LoRALayer(256, 128, rank=4)
        
        # Freeze original layer
        for param in linear.parameters():
            param.requires_grad = False
        
        # Test combined forward pass
        input_tensor = torch.randn(8, 256)
        
        # Original output
        original_output = linear(input_tensor)
        
        # LoRA adaptation
        lora_adaptation = lora(input_tensor)
        
        # Combined output
        combined_output = original_output + lora_adaptation
        
        assert combined_output.shape == (8, 128)
        assert not torch.allclose(combined_output, original_output)
    
    def test_parameter_efficiency_constraint(self):
        """Test that adaptations respect parameter efficiency constraints"""
        config = AdaptationConfig(max_trainable_ratio=0.02)
        
        # Simulate a large model
        original_params = 100_000_000  # 100M parameters
        max_trainable = int(original_params * config.max_trainable_ratio)
        
        # LoRA with rank 4
        in_features = 1000
        out_features = 1000
        lora = LoRALayer(in_features, out_features, rank=config.rank)
        
        lora_params = config.rank * (in_features + out_features)
        efficiency_ratio = lora_params / (in_features * out_features)
        
        assert efficiency_ratio < config.max_trainable_ratio
    
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory efficiency of LoRA adaptation"""
        # Large layer simulation
        large_linear = nn.Linear(4096, 4096)
        lora_adapter = LoRALayer(4096, 4096, rank=16)
        
        # Count parameters
        original_params = sum(p.numel() for p in large_linear.parameters())
        lora_params = sum(p.numel() for p in lora_adapter.parameters())
        
        # Memory efficiency check
        efficiency = lora_params / original_params
        assert efficiency < 0.01  # Less than 1% of original parameters
        
        # Performance check - forward pass should work
        input_tensor = torch.randn(4, 4096)
        
        # Freeze original
        for param in large_linear.parameters():
            param.requires_grad = False
        
        # Combined forward pass
        original_out = large_linear(input_tensor)
        lora_out = lora_adapter(input_tensor)
        combined = original_out + lora_out
        
        assert combined.shape == (4, 4096)


@pytest.mark.model
class TestModelCompatibility:
    """Test compatibility with various model architectures"""
    
    def test_resnet_compatibility(self):
        """Test LoRA compatibility with ResNet-like architectures"""
        # Simulate ResNet block
        conv_layer = nn.Conv2d(64, 64, 3, padding=1)
        
        # For conv layers, we'd typically adapt the following linear layers
        # Here we test the concept with a linear projection
        linear_proj = nn.Linear(64, 64)
        lora_adaptation = LoRALayer(64, 64, rank=4)
        
        # Test forward pass
        input_tensor = torch.randn(1, 64)
        
        original_out = linear_proj(input_tensor)
        lora_out = lora_adaptation(input_tensor)
        combined_out = original_out + lora_out
        
        assert combined_out.shape == (1, 64)
    
    def test_transformer_compatibility(self):
        """Test LoRA compatibility with Transformer-like architectures"""
        # Simulate attention layer dimensions
        embed_dim = 768
        
        # Query, Key, Value projections
        q_proj = nn.Linear(embed_dim, embed_dim)
        k_proj = nn.Linear(embed_dim, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim)
        
        # LoRA adaptations
        q_lora = LoRALayer(embed_dim, embed_dim, rank=8)
        k_lora = LoRALayer(embed_dim, embed_dim, rank=8)
        v_lora = LoRALayer(embed_dim, embed_dim, rank=8)
        
        # Test forward pass
        input_tensor = torch.randn(4, 10, embed_dim)  # [batch, seq_len, embed_dim]
        
        # Reshape for linear layers
        batch_size, seq_len, embed_dim = input_tensor.shape
        input_flat = input_tensor.view(-1, embed_dim)
        
        # Adapted projections
        q = q_proj(input_flat) + q_lora(input_flat)
        k = k_proj(input_flat) + k_lora(input_flat)
        v = v_proj(input_flat) + v_lora(input_flat)
        
        # Reshape back
        q = q.view(batch_size, seq_len, embed_dim)
        k = k.view(batch_size, seq_len, embed_dim)
        v = v.view(batch_size, seq_len, embed_dim)
        
        assert q.shape == input_tensor.shape
        assert k.shape == input_tensor.shape
        assert v.shape == input_tensor.shape


if __name__ == "__main__":
    pytest.main([__file__])
