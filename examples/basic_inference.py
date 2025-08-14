#!/usr/bin/env python3
"""
Basic inference example for PCB defect detection.

This script demonstrates how to:
1. Load a pre-trained model
2. Process a single PCB image
3. Get predictions with confidence scores
4. Save results

Usage:
    python examples/basic_inference.py --image path/to/pcb.jpg
    python examples/basic_inference.py --image path/to/pcb.jpg --model checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to Python path
import sys
sys.path.append('.')

from core.foundation_adapter import FoundationAdapter


def load_model(model_path: str = None, backbone: str = 'resnet50') -> FoundationAdapter:
    """Load pre-trained model or initialize new one."""
    print(f"Loading model (backbone: {backbone})...")
    
    model = FoundationAdapter(
        backbone=backbone,
        num_classes=6,  # Adjust based on your dataset
        lora_rank=16,
        lora_alpha=32
    )
    
    if model_path and Path(model_path).exists():
        print(f"Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print("Using randomly initialized model (for demonstration)")
        print("Note: Download or train a model for real predictions")
    
    model.eval()
    return model


def predict_image(model: FoundationAdapter, image_path: str) -> Dict[str, Any]:
    """Predict defects in a single image."""
    print(f"Processing image: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Make prediction
    with torch.no_grad():
        prediction = model.predict(image)
    
    return prediction


def visualize_results(image_path: str, prediction: Dict[str, Any], output_path: str = None):
    """Visualize prediction results."""
    # Load original image
    image = Image.open(image_path)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')
    
    # Add prediction text
    pred_text = f"Prediction: {prediction['class']}\nConfidence: {prediction['confidence']:.3f}"
    ax.text(10, 30, pred_text, fontsize=14, color='white', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.title(f"PCB Defect Detection Result", fontsize=16)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(prediction: Dict[str, Any], image_path: str, output_file: str):
    """Save prediction results to JSON file."""
    results = {
        'image_path': str(image_path),
        'prediction': prediction,
        'model_info': {
            'backbone': 'resnet50',
            'version': '1.0.0'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="PCB Defect Detection - Basic Inference")
    parser.add_argument('--image', '-i', required=True, help="Path to PCB image")
    parser.add_argument('--model', '-m', help="Path to model checkpoint")
    parser.add_argument('--backbone', default='resnet50', help="Model backbone (resnet50, clip)")
    parser.add_argument('--output-dir', '-o', default='outputs', help="Output directory")
    parser.add_argument('--visualize', action='store_true', help="Create visualization")
    parser.add_argument('--save-results', action='store_true', help="Save results to JSON")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load model
        model = load_model(args.model, args.backbone)
        
        # Make prediction
        prediction = predict_image(model, args.image)
        
        # Print results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"Predicted Class: {prediction['class']}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        
        if 'probabilities' in prediction:
            print(f"All Probabilities:")
            for class_name, prob in prediction['probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
        
        # Optional: Create visualization
        if args.visualize:
            image_name = Path(args.image).stem
            viz_path = output_dir / f"{image_name}_prediction.png"
            visualize_results(args.image, prediction, str(viz_path))
        
        # Optional: Save results
        if args.save_results:
            image_name = Path(args.image).stem
            results_path = output_dir / f"{image_name}_results.json"
            save_results(prediction, args.image, str(results_path))
        
        print("\n Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have the correct model and image paths.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
