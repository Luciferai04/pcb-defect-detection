"""
Ensemble Prediction with Uncertainty Quantification
Implements multiple ensemble methods and uncertainty estimation techniques for robust PCB defect detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    ensemble_size: int = 5
    dropout_samples: int = 50
    temperature: float = 1.0
    calibration_samples: int = 1000
    uncertainty_threshold: float = 0.1
    
class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Add dropout layers after each attention layer
        self._inject_mc_dropout()
    
    def _inject_mc_dropout(self):
        """Inject MC Dropout layers into the model"""
        for name, module in self.base_model.named_modules():
            if 'attention' in name.lower() or 'mlp' in name.lower():
                # Add MC dropout after attention/MLP layers
                if hasattr(module, 'dropout'):
                    module.dropout = nn.Dropout(p=self.dropout_rate)
    
    def forward(self, x: torch.Tensor, num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with MC Dropout sampling"""
        self.train()  # Enable dropout during inference
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.base_model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return mean_pred, std_pred

class DeepEnsemble(nn.Module):
    """Deep Ensemble for uncertainty quantification"""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble forward pass"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        
        # Compute epistemic uncertainty (model disagreement)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Compute aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = torch.mean(predictions * (1 - predictions), dim=0)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_pred, total_uncertainty

class BayesianNeuralNetwork(nn.Module):
    """Variational Bayesian Neural Network layer"""
    
    def __init__(self, in_features: int, out_features: int, prior_mu: float = 0.0, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Prior parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        self.weight_mu.data.normal_(0, 0.1)
        self.weight_rho.data.fill_(-3)  # rho = log(sigma)
        self.bias_mu.data.normal_(0, 0.1)
        self.bias_rho.data.fill_(-3)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with weight sampling"""
        if sample:
            # Sample weights and biases
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # Weight KL divergence
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_kl = self._kl_divergence(self.weight_mu, weight_sigma, self.prior_mu, self.prior_sigma)
        
        # Bias KL divergence
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        bias_kl = self._kl_divergence(self.bias_mu, bias_sigma, self.prior_mu, self.prior_sigma)
        
        return weight_kl + bias_kl
    
    def _kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):
        """Compute KL divergence between two Gaussians"""
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        return kl.sum()

class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification for PCB defect detection"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Calibration data for temperature scaling
        self.calibration_data = None
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
    
    def calibrate_temperature(self, logits: torch.Tensor, labels: torch.Tensor):
        """Temperature scaling for calibration"""
        self.calibration_data = (logits, labels)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.logger.info(f"Calibrated temperature: {self.temperature.item():.3f}")
    
    def compute_predictive_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute predictive entropy"""
        probs = F.softmax(predictions, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return entropy
    
    def compute_mutual_information(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute mutual information between predictions and model parameters"""
        # Mean prediction across samples
        mean_pred = torch.mean(predictions, dim=0)
        mean_entropy = self.compute_predictive_entropy(mean_pred)
        
        # Expected entropy
        sample_entropies = torch.stack([
            self.compute_predictive_entropy(pred) for pred in predictions
        ])
        expected_entropy = torch.mean(sample_entropies, dim=0)
        
        # Mutual information = mean entropy - expected entropy
        mutual_info = mean_entropy - expected_entropy
        return mutual_info
    
    def detect_out_of_distribution(self, 
                                   predictions: torch.Tensor, 
                                   uncertainty: torch.Tensor, 
                                   threshold: Optional[float] = None) -> torch.Tensor:
        """Detect out-of-distribution samples based on uncertainty"""
        if threshold is None:
            threshold = self.config.uncertainty_threshold
        
        # Normalize uncertainty to [0, 1]
        normalized_uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        
        ood_mask = normalized_uncertainty > threshold
        return ood_mask
    
    def compute_confidence_intervals(self, 
                                     predictions: torch.Tensor, 
                                     confidence_level: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute confidence intervals for predictions"""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = torch.quantile(predictions, lower_percentile / 100, dim=0)
        upper_bound = torch.quantile(predictions, upper_percentile / 100, dim=0)
        
        return lower_bound, upper_bound

class EnsemblePredictor:
    """Main ensemble predictor with comprehensive uncertainty quantification"""
    
    def __init__(self, 
                 models: List[nn.Module], 
                 config: EnsembleConfig,
                 device: str = 'cuda'):
        self.models = models
        self.config = config
        self.device = device
        self.uncertainty_quantifier = UncertaintyQuantifier(config)
        
        # Move models to device
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict_ensemble(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Comprehensive ensemble prediction with uncertainty quantification"""
        x = x.to(self.device)
        
        # Deep ensemble predictions
        deep_ensemble = DeepEnsemble(self.models)
        ensemble_mean, ensemble_uncertainty = deep_ensemble(x)
        
        # Monte Carlo Dropout predictions
        mc_predictions = []
        for model in self.models:
            mc_model = MonteCarloDropout(model)
            mc_mean, mc_std = mc_model(x, num_samples=self.config.dropout_samples)
            mc_predictions.append(mc_mean)
        
        mc_predictions = torch.stack(mc_predictions)
        mc_ensemble_mean = torch.mean(mc_predictions, dim=0)
        mc_ensemble_std = torch.std(mc_predictions, dim=0)
        
        # Predictive entropy
        predictive_entropy = self.uncertainty_quantifier.compute_predictive_entropy(ensemble_mean)
        
        # Mutual information
        mutual_info = self.uncertainty_quantifier.compute_mutual_information(mc_predictions)
        
        # Confidence intervals
        lower_bound, upper_bound = self.uncertainty_quantifier.compute_confidence_intervals(mc_predictions)
        
        # Out-of-distribution detection
        ood_mask = self.uncertainty_quantifier.detect_out_of_distribution(
            ensemble_mean, ensemble_uncertainty
        )
        
        results = {
            'predictions': ensemble_mean,
            'epistemic_uncertainty': ensemble_uncertainty,
            'aleatoric_uncertainty': mc_ensemble_std,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_info,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'ood_detection': ood_mask,
            'mc_predictions': mc_predictions
        }
        
        return results
    
    def evaluate_calibration(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model calibration"""
        # Expected Calibration Error (ECE)
        confidences = torch.max(F.softmax(predictions, dim=-1), dim=-1)[0]
        accuracies = (torch.argmax(predictions, dim=-1) == labels).float()
        
        bin_boundaries = torch.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
            
            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = torch.max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'expected_calibration_error': ece.item(),
            'maximum_calibration_error': mce.item()
        }

class AdaptiveEnsemble:
    """Adaptive ensemble that adjusts based on input characteristics"""
    
    def __init__(self, models: List[nn.Module], config: EnsembleConfig):
        self.models = models
        self.config = config
        self.model_weights = torch.ones(len(models)) / len(models)
        
        # Meta-learner to predict model weights
        self.meta_learner = nn.Sequential(
            nn.Linear(2048, 512),  # Assuming 2048-dim features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(models)),
            nn.Softmax(dim=-1)
        )
    
    def extract_input_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input for meta-learning"""
        # Use first model's feature extractor
        with torch.no_grad():
            features = self.models[0].encode_image(x)
        return features
    
    def compute_adaptive_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weights based on input characteristics"""
        features = self.extract_input_features(x)
        weights = self.meta_learner(features)
        return weights
    
    def predict_adaptive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive ensemble prediction"""
        # Get adaptive weights
        weights = self.compute_adaptive_weights(x)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # [batch, models, classes]
        
        # Weighted ensemble
        weighted_pred = torch.sum(predictions * weights.unsqueeze(-1), dim=1)
        
        # Compute uncertainty as weighted variance
        uncertainty = torch.sum(weights.unsqueeze(-1) * (predictions - weighted_pred.unsqueeze(1))**2, dim=1)
        
        return weighted_pred, uncertainty

def create_ensemble_config() -> EnsembleConfig:
    """Create default ensemble configuration"""
    return EnsembleConfig(
        ensemble_size=5,
        dropout_samples=50,
        temperature=1.0,
        calibration_samples=1000,
        uncertainty_threshold=0.1
    )

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = create_ensemble_config()
    
    # Create dummy models for testing
    dummy_models = [
        nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 5)  # 5 defect classes
        ) for _ in range(config.ensemble_size)
    ]
    
    # Initialize ensemble predictor
    ensemble_predictor = EnsemblePredictor(dummy_models, config)
    
    # Test with dummy data
    test_input = torch.randn(16, 2048)  # Batch of 16 samples
    
    # Make predictions
    results = ensemble_predictor.predict_ensemble(test_input)
    
    print("Ensemble Prediction Results:")
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Epistemic uncertainty shape: {results['epistemic_uncertainty'].shape}")
    print(f"Predictive entropy shape: {results['predictive_entropy'].shape}")
    print(f"OOD detection: {results['ood_detection'].sum().item()} samples detected as OOD")
