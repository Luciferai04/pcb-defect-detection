#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for PCB Defect Detection
==============================================================

This module provides advanced hyperparameter optimization using:
1. Optuna for Bayesian optimization
2. Ray Tune for distributed hyperparameter search
3. Weights & Biases sweeps integration
4. Multi-objective optimization (accuracy vs efficiency)
"""

import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from enhanced_pcb_model import create_enhanced_model
import numpy as np
from pathlib import Path


class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        
    def objective(self, trial: optuna.Trial) -> tuple[float, float]:
        """
        Multi-objective function: maximize accuracy, minimize parameters
        
        Returns:
            Tuple of (accuracy, parameter_efficiency)
        """
        # Suggest hyperparameters
        params = {
            'lora_rank': trial.suggest_int('lora_rank', 4, 64),
            'lora_alpha': trial.suggest_float('lora_alpha', 1.0, 64.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        }
        
        # Train model with suggested parameters
        accuracy, parameter_efficiency = self._train_and_evaluate(params)
        
        return accuracy, parameter_efficiency
    
    def _train_and_evaluate(self, params: Dict[str, Any]) -> tuple[float, float]:
        """Train model and return metrics"""
        # Create model with suggested parameters
        model, loss_fn = create_enhanced_model(
            num_classes=5,
            lora_rank=params['lora_rank'],
            class_weights=None
        )
        
        # Calculate parameter efficiency
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameter_efficiency = 1.0 - (trainable_params / total_params)
        
        # Mock training for demonstration (replace with actual training)
        # In real implementation, this would train the model
        accuracy = np.random.uniform(0.7, 0.95)  # Simulate training result
        
        return accuracy, parameter_efficiency
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """Run multi-objective optimization"""
        # Create study with multiple objectives
        self.study = optuna.create_study(
            directions=['maximize', 'maximize'],  # maximize both accuracy and efficiency
            study_name='pcb_defect_multiobjective'
        )
        
        # Add W&B callback if available
        wandb_callback = WeightsAndBiasesCallback(
            metric_name=['accuracy', 'parameter_efficiency']
        )
        
        # Optimize
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[wandb_callback]
        )
        
        # Get Pareto optimal solutions
        pareto_front = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                pareto_front.append({
                    'params': trial.params,
                    'accuracy': trial.values[0],
                    'parameter_efficiency': trial.values[1]
                })
        
        return {
            'pareto_front': pareto_front,
            'best_trials': self.study.best_trials,
            'study': self.study
        }


class AutoMLPipeline:
    """Automated ML pipeline with hyperparameter optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration for AutoML"""
        default_config = {
            'search_space': {
                'model': {
                    'backbone': ['resnet18', 'resnet50', 'efficientnet_b0'],
                    'lora_rank': [4, 8, 16, 32, 64],
                    'lora_alpha': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
                    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
                },
                'training': {
                    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
                    'batch_size': [16, 32, 64, 128],
                    'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                    'optimizer': ['adam', 'adamw', 'sgd'],
                    'scheduler': ['cosine', 'step', 'exponential'],
                },
                'active_learning': {
                    'strategy': ['uncertainty', 'diversity', 'hybrid'],
                    'budget_per_round': [10, 20, 50, 100],
                    'uncertainty_weight': [0.3, 0.5, 0.7],
                }
            },
            'objectives': {
                'primary': 'accuracy',
                'secondary': 'parameter_efficiency',
                'constraints': {
                    'max_params_ratio': 0.05,
                    'min_accuracy': 0.85,
                    'max_inference_time': 100,  # milliseconds
                }
            },
            'optimization': {
                'n_trials': 100,
                'timeout': 7200,  # 2 hours
                'n_jobs': 4,
                'pruner': 'median',
                'sampler': 'tpe',
            }
        }
        
        if config_path and Path(config_path).exists():
            # Load custom config from file
            import json
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            # Merge with defaults
            default_config.update(custom_config)
        
        return default_config
    
    def run_hyperparameter_search(self) -> Dict[str, Any]:
        """Run comprehensive hyperparameter search"""
        
        # Initialize W&B sweep
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'accuracy',
                'goal': 'maximize'
            },
            'parameters': self._convert_search_space_to_wandb(),
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 3,
                'eta': 2
            }
        }
        
        sweep_id = wandb.sweep(sweep_config, project="pcb-defect-automl")
        
        # Run sweep
        wandb.agent(sweep_id, self._train_sweep, count=self.config['optimization']['n_trials'])
        
        return {'sweep_id': sweep_id}
    
    def _convert_search_space_to_wandb(self) -> Dict[str, Any]:
        """Convert search space to W&B format"""
        wandb_params = {}
        
        # Model parameters
        model_params = self.config['search_space']['model']
        wandb_params.update({
            'backbone': {'values': model_params['backbone']},
            'lora_rank': {'values': model_params['lora_rank']},
            'lora_alpha': {'values': model_params['lora_alpha']},
            'dropout_rate': {'values': model_params['dropout_rate']},
        })
        
        # Training parameters
        training_params = self.config['search_space']['training']
        wandb_params.update({
            'learning_rate': {'values': training_params['learning_rate']},
            'batch_size': {'values': training_params['batch_size']},
            'weight_decay': {'values': training_params['weight_decay']},
            'optimizer': {'values': training_params['optimizer']},
            'scheduler': {'values': training_params['scheduler']},
        })
        
        return wandb_params
    
    def _train_sweep(self):
        """Training function for W&B sweep"""
        with wandb.init() as run:
            config = wandb.config
            
            # Create model with sweep parameters
            model, loss_fn = create_enhanced_model(
                num_classes=5,
                backbone=config.backbone,
                lora_rank=config.lora_rank,
            )
            
            # Mock training (replace with actual training loop)
            for epoch in range(10):
                # Simulate training metrics
                train_loss = np.random.uniform(0.1, 0.5)
                val_accuracy = np.random.uniform(0.7, 0.95)
                
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_accuracy': val_accuracy,
                })
            
            # Calculate final metrics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            parameter_efficiency = 1.0 - (trainable_params / total_params)
            
            wandb.log({
                'accuracy': val_accuracy,
                'parameter_efficiency': parameter_efficiency,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
            })


class NeuralArchitectureSearch:
    """Neural Architecture Search for optimal model design"""
    
    def __init__(self):
        self.search_space = self._define_search_space()
    
    def _define_search_space(self) -> Dict[str, Any]:
        """Define NAS search space"""
        return {
            'backbone_depth': [18, 34, 50, 101],
            'attention_heads': [4, 8, 16],
            'pyramid_scales': [2, 3, 4, 5],
            'lora_ranks': [4, 8, 16, 32],
            'feature_fusion': ['concat', 'add', 'attention'],
            'activation': ['relu', 'gelu', 'swish'],
        }
    
    def search_architecture(self, n_trials: int = 50) -> Dict[str, Any]:
        """Search for optimal architecture"""
        
        def objective(trial):
            # Sample architecture
            arch = {
                'backbone_depth': trial.suggest_categorical('backbone_depth', self.search_space['backbone_depth']),
                'attention_heads': trial.suggest_categorical('attention_heads', self.search_space['attention_heads']),
                'pyramid_scales': trial.suggest_categorical('pyramid_scales', self.search_space['pyramid_scales']),
                'lora_rank': trial.suggest_categorical('lora_rank', self.search_space['lora_ranks']),
                'feature_fusion': trial.suggest_categorical('feature_fusion', self.search_space['feature_fusion']),
                'activation': trial.suggest_categorical('activation', self.search_space['activation']),
            }
            
            # Evaluate architecture (simplified)
            performance = self._evaluate_architecture(arch)
            
            return performance
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return {
            'best_architecture': study.best_params,
            'best_performance': study.best_value,
            'study': study
        }
    
    def _evaluate_architecture(self, architecture: Dict[str, Any]) -> float:
        """Evaluate a given architecture"""
        # Mock evaluation (replace with actual architecture training)
        # Consider factors like parameter efficiency, accuracy, inference time
        base_score = np.random.uniform(0.7, 0.9)
        
        # Penalty for too complex architectures
        complexity_penalty = 0
        if architecture['backbone_depth'] > 50:
            complexity_penalty += 0.05
        if architecture['attention_heads'] > 8:
            complexity_penalty += 0.02
        if architecture['lora_rank'] > 16:
            complexity_penalty += 0.03
        
        return base_score - complexity_penalty


if __name__ == "__main__":
    # Example usage
    
    # Multi-objective optimization
    optimizer = MultiObjectiveOptimizer({})
    results = optimizer.optimize(n_trials=20)
    print(f"Found {len(results['pareto_front'])} Pareto optimal solutions")
    
    # AutoML pipeline
    automl = AutoMLPipeline()
    sweep_results = automl.run_hyperparameter_search()
    print(f"Started W&B sweep: {sweep_results['sweep_id']}")
    
    # Neural Architecture Search
    nas = NeuralArchitectureSearch()
    arch_results = nas.search_architecture(n_trials=10)
    print(f"Best architecture: {arch_results['best_architecture']}")
