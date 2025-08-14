#!/usr/bin/env python3
"""
Evaluation Metrics for Foundation Model Adaptation
=================================================

Comprehensive evaluation framework implementing:
- Transfer Score metrics for unsupervised evaluation
- Cross-Dataset Consistency (CDC) for robustness assessment
- Corruption robustness testing with domain-specific perturbations
- Parameter efficiency vs accuracy trade-offs

Research Context:
- Unsupervised evaluation without target labels
- Domain shift robustness assessment
- Few-shot learning performance tracking
- Computational efficiency analysis


Reference: Latest research on transfer learning evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import clip
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation metrics.
    
    Defines thresholds and parameters for various evaluation metrics
    based on research findings for domain adaptation assessment.
    """
    # Transfer Score parameters
    confidence_threshold: float = 0.8
    entropy_threshold: float = 0.5
    
    # Cross-Dataset Consistency parameters
    consistency_samples: int = 1000
    bootstrap_iterations: int = 100
    
    # Corruption robustness parameters
    corruption_levels: List[float] = None
    corruption_types: List[str] = None
    
    # Parameter efficiency thresholds
    max_efficiency_ratio: float = 0.02
    min_accuracy_threshold: float = 0.7
    
    def __post_init__(self):
        if self.corruption_levels is None:
            self.corruption_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        if self.corruption_types is None:
            self.corruption_types = [
                'gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur', 'motion_blur',
                'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate', 'jpeg_compression'
            ]

class TransferScoreMetric:
    """
    Transfer Score: Unsupervised evaluation metric for domain adaptation.
    
    Measures adaptation quality without target domain labels by analyzing:
    - Prediction confidence distribution
    - Feature space organization
    - Cross-domain consistency
    
    Based on research showing correlation with supervised performance.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def compute_confidence_score(self, logits: torch.Tensor) -> float:
        """
        Compute confidence-based transfer score.
        
        High-quality adaptation should produce confident predictions
        on target domain data.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            
        Returns:
            Confidence score (0-1, higher is better)
        """
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Compute fraction of confident predictions
        confident_mask = max_probs > self.config.confidence_threshold
        confidence_ratio = confident_mask.float().mean().item()
        
        # Compute average confidence of confident predictions
        if confident_mask.sum() > 0:
            avg_confidence = max_probs[confident_mask].mean().item()
        else:
            avg_confidence = 0.0
        
        # Combined confidence score
        confidence_score = 0.7 * confidence_ratio + 0.3 * avg_confidence
        
        return confidence_score
    
    def compute_entropy_score(self, logits: torch.Tensor) -> float:
        """
        Compute entropy-based transfer score.
        
        Well-adapted models should have low entropy (sharp predictions)
        on target domain data.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            
        Returns:
            Entropy score (0-1, higher is better)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Normalize entropy by log(num_classes)
        max_entropy = np.log(logits.shape[-1])
        normalized_entropy = entropy / max_entropy
        
        # Convert to score (lower entropy = higher score)
        entropy_score = 1.0 - normalized_entropy.mean().item()
        
        return max(0.0, entropy_score)
    
    def compute_diversity_score(self, logits: torch.Tensor) -> float:
        """
        Compute prediction diversity score.
        
        Good adaptation should maintain class balance and avoid
        collapsed predictions to a single class.
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            
        Returns:
            Diversity score (0-1, higher is better)
        """
        predictions = torch.argmax(logits, dim=-1)
        num_classes = logits.shape[-1]
        
        # Compute class distribution
        class_counts = torch.bincount(predictions, minlength=num_classes)
        class_probs = class_counts.float() / len(predictions)
        
        # Compute entropy of class distribution
        class_probs = class_probs + 1e-8  # Avoid log(0)
        class_entropy = -(class_probs * torch.log(class_probs)).sum()
        
        # Normalize by maximum possible entropy
        max_class_entropy = np.log(num_classes)
        diversity_score = (class_entropy / max_class_entropy).item()
        
        return diversity_score
    
    def compute_transfer_score(self, source_logits: torch.Tensor,
                              target_logits: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive transfer score.
        
        Combines multiple unsupervised metrics to assess adaptation quality.
        
        Args:
            source_logits: Predictions on source domain data
            target_logits: Predictions on target domain data
            
        Returns:
            Dictionary of transfer score components
        """
        # Individual score components
        target_confidence = self.compute_confidence_score(target_logits)
        target_entropy = self.compute_entropy_score(target_logits)
        target_diversity = self.compute_diversity_score(target_logits)
        
        # Source domain retention (should maintain performance)
        source_confidence = self.compute_confidence_score(source_logits)
        retention_score = min(1.0, source_confidence / 0.8)  # Assume 0.8 baseline
        
        # Combined transfer score
        transfer_score = (
            0.4 * target_confidence +
            0.3 * target_entropy +
            0.2 * target_diversity +
            0.1 * retention_score
        )
        
        return {
            'transfer_score': transfer_score,
            'target_confidence': target_confidence,
            'target_entropy': target_entropy,
            'target_diversity': target_diversity,
            'source_retention': retention_score
        }

class CrossDatasetConsistency:
    """
    Cross-Dataset Consistency (CDC): Robustness evaluation across datasets.
    
    Measures how consistently a model performs across different datasets
    from the same domain, indicating robustness to dataset-specific biases.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def compute_consistency_score(self, predictions_list: List[torch.Tensor],
                                 labels_list: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute cross-dataset consistency score.
        
        Args:
            predictions_list: List of prediction tensors from different datasets
            labels_list: List of corresponding label tensors
            
        Returns:
            Dictionary of consistency metrics
        """
        if len(predictions_list) < 2:
            raise ValueError("Need at least 2 datasets for consistency evaluation")
        
        # Compute accuracy for each dataset
        accuracies = []
        for preds, labels in zip(predictions_list, labels_list):
            pred_classes = torch.argmax(preds, dim=-1)
            accuracy = (pred_classes == labels).float().mean().item()
            accuracies.append(accuracy)
        
        # Consistency metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        min_accuracy = np.min(accuracies)
        max_accuracy = np.max(accuracies)
        
        # Consistency score (lower variance = higher consistency)
        consistency_score = 1.0 - (std_accuracy / (mean_accuracy + 1e-8))
        consistency_score = max(0.0, consistency_score)
        
        return {
            'consistency_score': consistency_score,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'accuracy_range': max_accuracy - min_accuracy
        }
    
    def compute_feature_consistency(self, features_list: List[torch.Tensor],
                                   labels_list: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compute feature-level consistency across datasets.
        
        Measures how similar the learned feature representations are
        across different datasets.
        
        Args:
            features_list: List of feature tensors from different datasets
            labels_list: List of corresponding label tensors
            
        Returns:
            Dictionary of feature consistency metrics
        """
        # Compute class centroids for each dataset
        centroids_list = []
        for features, labels in zip(features_list, labels_list):
            unique_labels = torch.unique(labels)
            centroids = []
            
            for label in unique_labels:
                mask = labels == label
                if mask.sum() > 0:
                    centroid = features[mask].mean(dim=0)
                    centroids.append(centroid)
            
            if centroids:
                centroids = torch.stack(centroids)
                centroids_list.append(centroids)
        
        if len(centroids_list) < 2:
            return {'feature_consistency': 0.0}
        
        # Compute pairwise cosine similarities between centroids
        similarities = []
        for i in range(len(centroids_list)):
            for j in range(i + 1, len(centroids_list)):
                centroids_i = F.normalize(centroids_list[i], dim=-1)
                centroids_j = F.normalize(centroids_list[j], dim=-1)
                
                # Compute similarity matrix
                sim_matrix = centroids_i @ centroids_j.t()
                
                # Take diagonal (same class similarities)
                if sim_matrix.shape[0] == sim_matrix.shape[1]:
                    similarities.extend(torch.diag(sim_matrix).tolist())
        
        if not similarities:
            return {'feature_consistency': 0.0}
        
        # Average similarity as consistency score
        feature_consistency = np.mean(similarities)
        
        return {
            'feature_consistency': feature_consistency,
            'similarity_std': np.std(similarities),
            'num_comparisons': len(similarities)
        }

class CorruptionRobustness:
    """
    Corruption robustness evaluation with domain-specific perturbations.
    
    Tests model robustness to various types of corruptions that commonly
    occur in specialized domains (medical, remote sensing, etc.).
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def apply_gaussian_noise(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """Apply Gaussian noise corruption."""
        noise = torch.randn_like(images) * severity * 0.15
        return torch.clamp(images + noise, 0, 1)
    
    def apply_shot_noise(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """Apply shot noise corruption."""
        noise = torch.poisson(images * severity * 10) / (severity * 10)
        return torch.clamp(noise, 0, 1)
    
    def apply_defocus_blur(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """Apply defocus blur corruption."""
        # Simplified blur using average pooling
        kernel_size = int(2 * severity + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blur = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
        return blur(images)
    
    def apply_brightness(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """Apply brightness corruption."""
        brightness_factor = 1.0 + severity * 0.5
        return torch.clamp(images * brightness_factor, 0, 1)
    
    def apply_contrast(self, images: torch.Tensor, severity: float) -> torch.Tensor:
        """Apply contrast corruption."""
        mean_img = images.mean(dim=[2, 3], keepdim=True)
        contrast_factor = 1.0 + severity
        return torch.clamp(mean_img + contrast_factor * (images - mean_img), 0, 1)
    
    def apply_corruption(self, images: torch.Tensor, corruption_type: str,
                        severity: float) -> torch.Tensor:
        """
        Apply specified corruption to images.
        
        Args:
            images: Input images [batch_size, 3, H, W]
            corruption_type: Type of corruption to apply
            severity: Severity level (0-1)
            
        Returns:
            Corrupted images
        """
        if corruption_type == 'gaussian_noise':
            return self.apply_gaussian_noise(images, severity)
        elif corruption_type == 'shot_noise':
            return self.apply_shot_noise(images, severity)
        elif corruption_type == 'defocus_blur':
            return self.apply_defocus_blur(images, severity)
        elif corruption_type == 'brightness':
            return self.apply_brightness(images, severity)
        elif corruption_type == 'contrast':
            return self.apply_contrast(images, severity)
        else:
            logger.warning(f"Unknown corruption type: {corruption_type}")
            return images
    
    def evaluate_robustness(self, model, images: torch.Tensor, labels: torch.Tensor,
                           domain_id: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness to various corruptions.
        
        Args:
            model: Model to evaluate
            images: Clean images
            labels: Ground truth labels
            domain_id: Domain identifier
            
        Returns:
            Dictionary of robustness metrics for each corruption
        """
        model.eval()
        results = {}
        
        # Evaluate clean performance
        with torch.no_grad():
            if hasattr(model, 'forward'):
                clean_outputs = model(images, domain_id)
                if isinstance(clean_outputs, dict):
                    clean_logits = clean_outputs.get('logits', clean_outputs.get('logits_per_image'))
                else:
                    clean_logits = clean_outputs
            else:
                clean_logits = model(images)
            
            clean_preds = torch.argmax(clean_logits, dim=-1)
            clean_accuracy = (clean_preds == labels).float().mean().item()
        
        results['clean'] = {'accuracy': clean_accuracy}
        
        # Evaluate robustness to each corruption type
        for corruption_type in self.config.corruption_types[:5]:  # Limit for demo
            corruption_results = []
            
            for severity in self.config.corruption_levels:
                # Apply corruption
                corrupted_images = self.apply_corruption(images, corruption_type, severity)
                
                # Evaluate on corrupted images
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        outputs = model(corrupted_images, domain_id)
                        if isinstance(outputs, dict):
                            logits = outputs.get('logits', outputs.get('logits_per_image'))
                        else:
                            logits = outputs
                    else:
                        logits = model(corrupted_images)
                    
                    preds = torch.argmax(logits, dim=-1)
                    accuracy = (preds == labels).float().mean().item()
                
                corruption_results.append(accuracy)
            
            # Compute corruption metrics
            mean_accuracy = np.mean(corruption_results)
            relative_performance = mean_accuracy / clean_accuracy if clean_accuracy > 0 else 0
            
            results[corruption_type] = {
                'accuracies': corruption_results,
                'mean_accuracy': mean_accuracy,
                'relative_performance': relative_performance
            }
        
        return results

class CalibrationMetrics:
    """
    Calibration metrics for evaluating prediction confidence reliability.
    
    Includes Expected Calibration Error (ECE) and reliability diagrams
    to assess how well predicted probabilities match actual accuracy.
    """
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
    
    def compute_expected_calibration_error(self, probs: torch.Tensor, 
                                         labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted confidence and
        actual accuracy across confidence bins.
        
        Args:
            probs: [num_samples, num_classes] predicted probabilities
            labels: [num_samples] true class indices
            
        Returns:
            Dictionary with ECE and reliability metrics
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels).float()
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in current bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # ECE contribution from this bin
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                reliability_data.append({
                    'bin_lower': bin_lower.item(),
                    'bin_upper': bin_upper.item(),
                    'confidence': avg_confidence_in_bin.item(),
                    'accuracy': accuracy_in_bin.item(),
                    'proportion': prop_in_bin.item()
                })
        
        # Additional calibration metrics
        max_calibration_error = max([abs(data['confidence'] - data['accuracy']) 
                                   for data in reliability_data]) if reliability_data else 0.0
        
        # Overconfidence/underconfidence
        conf_acc_diff = [(data['confidence'] - data['accuracy']) * data['proportion'] 
                        for data in reliability_data]
        avg_calibration = sum(conf_acc_diff) if conf_acc_diff else 0.0
        
        return {
            'expected_calibration_error': ece.item(),
            'max_calibration_error': max_calibration_error,
            'average_calibration': avg_calibration,
            'reliability_data': reliability_data,
            'is_overconfident': avg_calibration > 0,
            'calibration_quality': 1.0 - ece.item()  # Higher is better
        }
    
    def compute_uncertainty_metrics(self, probs: torch.Tensor, 
                                  labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute uncertainty quantification metrics.
        
        Args:
            probs: [num_samples, num_classes] predicted probabilities
            labels: [num_samples] true class indices
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Predictive entropy (epistemic uncertainty)
        log_probs = torch.log(probs + 1e-12)
        predictive_entropy = -(probs * log_probs).sum(dim=-1)
        
        # Confidence (max probability)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Correctness
        predictions = torch.argmax(probs, dim=-1)
        correct = (predictions == labels).float()
        
        # Uncertainty-accuracy correlation
        entropy_correct_corr = torch.corrcoef(torch.stack([predictive_entropy, correct]))[0, 1]
        confidence_correct_corr = torch.corrcoef(torch.stack([max_probs, correct]))[0, 1]
        
        return {
            'mean_predictive_entropy': predictive_entropy.mean().item(),
            'mean_confidence': max_probs.mean().item(),
            'entropy_accuracy_correlation': entropy_correct_corr.item() if not torch.isnan(entropy_correct_corr) else 0.0,
            'confidence_accuracy_correlation': confidence_correct_corr.item() if not torch.isnan(confidence_correct_corr) else 0.0,
            'high_confidence_accuracy': correct[max_probs > 0.9].mean().item() if (max_probs > 0.9).sum() > 0 else 0.0,
            'low_confidence_accuracy': correct[max_probs < 0.6].mean().item() if (max_probs < 0.6).sum() > 0 else 0.0
        }
    
    def generate_reliability_diagram(self, probs: torch.Tensor, labels: torch.Tensor, 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate reliability diagram for calibration visualization.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        ece_results = self.compute_expected_calibration_error(probs, labels)
        reliability_data = ece_results['reliability_data']
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if reliability_data:
            bin_centers = [(data['bin_lower'] + data['bin_upper']) / 2 for data in reliability_data]
            confidences = [data['confidence'] for data in reliability_data]
            accuracies = [data['accuracy'] for data in reliability_data]
            proportions = [data['proportion'] for data in reliability_data]
            
            # Reliability diagram
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            bars = ax.bar(bin_centers, accuracies, width=1/self.n_bins, alpha=0.7, 
                         color='skyblue', edgecolor='black', linewidth=1)
            
            # Color bars by proportion
            for bar, prop in zip(bars, proportions):
                bar.set_alpha(0.3 + 0.7 * prop * 10)  # More samples = more opaque
            
            ax.plot(bin_centers, confidences, 'ro-', linewidth=2, markersize=6, 
                   label='Average Confidence')
            
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Reliability Diagram (ECE: {ece_results["expected_calibration_error"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class ParameterEfficiencyAnalyzer:
    """
    Analyzer for parameter efficiency vs accuracy trade-offs.
    
    Evaluates how well models balance parameter efficiency with
    adaptation performance across different domains.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def compute_efficiency_metrics(self, model) -> Dict[str, float]:
        """
        Compute parameter efficiency metrics.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary of efficiency metrics
        """
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
        
        efficiency_ratio = trainable_params / total_params if total_params > 0 else 0
        
        # Efficiency score based on research constraints (<2% optimal)
        if efficiency_ratio <= self.config.max_efficiency_ratio:
            efficiency_score = 1.0
        else:
            # Penalize higher ratios
            penalty = (efficiency_ratio - self.config.max_efficiency_ratio) * 10
            efficiency_score = max(0.0, 1.0 - penalty)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_score': efficiency_score,
            'meets_constraint': efficiency_ratio <= self.config.max_efficiency_ratio
        }
    
    def compute_pareto_efficiency(self, models_data: List[Dict[str, float]]) -> Dict[int, bool]:
        """
        Compute Pareto efficiency for accuracy-efficiency trade-off.
        
        Args:
            models_data: List of dicts with 'accuracy' and 'efficiency_ratio' keys
            
        Returns:
            Dictionary mapping model index to Pareto efficiency boolean
        """
        pareto_efficient = {}
        
        for i, model_i in enumerate(models_data):
            is_pareto = True
            
            for j, model_j in enumerate(models_data):
                if i != j:
                    # Check if model_j dominates model_i
                    if (model_j['accuracy'] >= model_i['accuracy'] and
                        model_j['efficiency_ratio'] <= model_i['efficiency_ratio'] and
                        (model_j['accuracy'] > model_i['accuracy'] or
                         model_j['efficiency_ratio'] < model_i['efficiency_ratio'])):
                        is_pareto = False
                        break
            
            pareto_efficient[i] = is_pareto
        
        return pareto_efficient

def evaluate_adaptation(model, source_loader, target_loader, domain_name: str,
                       config: EvaluationConfig = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation including:
    - Source domain retention
    - Target domain adaptation  
    - Cross-dataset consistency
    - Parameter efficiency metrics
    - Calibration and uncertainty metrics
    
    Args:
        model: Adapted model to evaluate
        source_loader: DataLoader for source domain data
        target_loader: DataLoader for target domain data
        domain_name: Name of the target domain
        config: Evaluation configuration
        
    Returns:
        Dictionary of comprehensive evaluation metrics
    """
    if config is None:
        config = EvaluationConfig()
    
    # Initialize evaluators
    transfer_score = TransferScoreMetric(config)
    corruption_eval = CorruptionRobustness(config)
    efficiency_analyzer = ParameterEfficiencyAnalyzer(config)
    calibration_metrics = CalibrationMetrics()
    
    results = {
        'domain': domain_name,
        'evaluation_config': config
    }
    
    # Parameter efficiency analysis
    efficiency_metrics = efficiency_analyzer.compute_efficiency_metrics(model)
    results['parameter_efficiency'] = efficiency_metrics
    
    # Collect predictions for transfer score computation
    model.eval()
    source_logits_list = []
    source_labels_list = []
    target_logits_list = []
    target_labels_list = []
    
    # Evaluate on source domain
    with torch.no_grad():
        for batch in source_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = batch[1]
            
            # Get model predictions
            if hasattr(model, 'forward'):
                outputs = model(images, domain_id=0)  # Source domain
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('logits_per_image'))
                else:
                    logits = outputs
            else:
                logits = model(images)
            
            source_logits_list.append(logits)
            source_labels_list.append(labels)
    
    # Evaluate on target domain
    with torch.no_grad():
        for batch in target_loader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images = batch[0]
                labels = batch[1]
            
            # Get model predictions
            if hasattr(model, 'forward'):
                outputs = model(images, domain_id=1)  # Target domain
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('logits_per_image'))
                else:
                    logits = outputs
            else:
                logits = model(images)
            
            target_logits_list.append(logits)
            target_labels_list.append(labels)
    
    # Concatenate all predictions
    if source_logits_list:
        source_logits = torch.cat(source_logits_list, dim=0)
        source_labels = torch.cat(source_labels_list, dim=0)
        
        # Source domain accuracy
        source_preds = torch.argmax(source_logits, dim=-1)
        source_accuracy = (source_preds == source_labels).float().mean().item()
        results['source_accuracy'] = source_accuracy
    
    if target_logits_list:
        target_logits = torch.cat(target_logits_list, dim=0)
        target_labels = torch.cat(target_labels_list, dim=0)
        
        # Target domain accuracy
        target_preds = torch.argmax(target_logits, dim=-1)
        target_accuracy = (target_preds == target_labels).float().mean().item()
        results['target_accuracy'] = target_accuracy
        
        # Calibration metrics on target domain
        target_probs = F.softmax(target_logits, dim=-1)
        calibration_results = calibration_metrics.compute_expected_calibration_error(
            target_probs, target_labels
        )
        uncertainty_results = calibration_metrics.compute_uncertainty_metrics(
            target_probs, target_labels
        )
        results['calibration_metrics'] = calibration_results
        results['uncertainty_metrics'] = uncertainty_results
        
        # Transfer score computation
        if source_logits_list:
            transfer_metrics = transfer_score.compute_transfer_score(
                source_logits[:1000],  # Limit for efficiency
                target_logits[:1000]
            )
            results['transfer_score'] = transfer_metrics
        
        # Corruption robustness (on subset for efficiency)
        subset_size = min(100, len(target_logits))
        subset_indices = torch.randperm(len(target_logits))[:subset_size]
        
        robustness_results = corruption_eval.evaluate_robustness(
            model,
            target_loader.dataset[subset_indices][0] if hasattr(target_loader.dataset, '__getitem__') else target_logits[:subset_size],
            target_labels[subset_indices],
            domain_id=1
        )
        results['corruption_robustness'] = robustness_results
    
    # Overall evaluation score
    overall_score = 0.0
    if 'target_accuracy' in results:
        overall_score += results['target_accuracy'] * 0.4
    if 'transfer_score' in results:
        overall_score += results['transfer_score']['transfer_score'] * 0.3
    if 'parameter_efficiency' in results:
        overall_score += results['parameter_efficiency']['efficiency_score'] * 0.3
    
    results['overall_score'] = overall_score
    
    logger.info(f"Evaluation complete for {domain_name}")
    logger.info(f"Target accuracy: {results.get('target_accuracy', 'N/A'):.4f}")
    logger.info(f"Parameter efficiency: {efficiency_metrics['efficiency_ratio']:.4f}")
    logger.info(f"Overall score: {overall_score:.4f}")
    
    return results

if __name__ == "__main__":
    # Demonstration of evaluation metrics
    print(" Foundation Model Adaptation Evaluation")
    print("=" * 50)
    
    # Create dummy data for demonstration
    batch_size, num_classes = 32, 5
    dummy_logits = torch.randn(batch_size, num_classes)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    
    # Initialize evaluators
    config = EvaluationConfig()
    transfer_metric = TransferScoreMetric(config)
    
    # Compute transfer score
    source_logits = torch.randn(batch_size, num_classes)
    target_logits = torch.randn(batch_size, num_classes)
    
    transfer_results = transfer_metric.compute_transfer_score(source_logits, target_logits)
    
    print(" Transfer Score Results:")
    for key, value in transfer_results.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n Evaluation framework ready!")
    print(f" Metrics: Transfer Score, CDC, Corruption Robustness")
    print(f" Focus: Unsupervised evaluation, parameter efficiency")
