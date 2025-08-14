#!/usr/bin/env python3
"""
PCB Defect Detection Inference Module
=====================================

Command-line interface for running PCB defect detection inference.
Supports single image inference, batch processing, and various output formats.

Usage:
    python -m api.inference --image path/to/image.jpg
    python -m api.inference --batch path/to/images/ --output results.json
    python -m api.inference --image image.jpg --model custom_model.pth --device cuda
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCBDefectInference:
    """Main inference class for PCB defect detection."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to trained model weights
            device: Target device ('cpu', 'cuda', 'mps', or 'auto')
            batch_size: Batch size for processing multiple images
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        
        # Class definitions
        self.class_names = [
            'normal',
            'missing_component', 
            'solder_bridge',
            'misalignment',
            'short_circuit'
        ]
        
        self.class_descriptions = {
            'normal': 'PCB board with no visible defects',
            'missing_component': 'PCB missing one or more electronic components',
            'solder_bridge': 'Unwanted solder connection between traces or components',
            'misalignment': 'Components not properly aligned on the PCB',
            'short_circuit': 'Electrical short circuit detected on PCB'
        }
        
        # Initialize model and preprocessing
        self._load_model()
        self._setup_preprocessing()
        
        logger.info(f"PCB Defect Detection Inference initialized on {self.device}")
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Determine the optimal device for inference."""
        if device == 'auto' or device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load the PCB defect detection model."""
        try:
            # Try to load enhanced PCB model
            from enhanced_pcb_model import create_enhanced_model
            self.model, _ = create_enhanced_model(num_classes=5)
            
            # Load pre-trained weights if provided
            if self.model_path and Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights from {self.model_path}")
            else:
                logger.warning("No model weights loaded - using initialized model")
            
        except ImportError:
            # Fallback to simple model
            logger.warning("Enhanced model not available, using simple fallback")
            import torch.nn as nn
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * 224 * 224, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(self.class_names))
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Calculate model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")
    
    def _setup_preprocessing(self):
        """Setup image preprocessing pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Image preprocessing pipeline initialized")
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise
    
    def predict_single(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        # Preprocess image
        try:
            image_tensor = self.preprocess_image(image_path).to(self.device)
        except Exception as e:
            return {
                'error': f"Failed to load image: {e}",
                'image_path': str(image_path)
            }
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
        
        inference_time = time.time() - start_time
        
        # Prepare results
        result = {
            'image_path': str(image_path),
            'predicted_class': self.class_names[predicted_class_id],
            'class_id': predicted_class_id,
            'confidence': float(confidence),
            'description': self.class_descriptions[self.class_names[predicted_class_id]],
            'probabilities': {
                self.class_names[i]: float(probabilities[0][i]) 
                for i in range(len(self.class_names))
            },
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_results = []
            
            for image_path in batch_paths:
                result = self.predict_single(image_path)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Progress logging
            processed = min(i + self.batch_size, len(image_paths))
            logger.info(f"Processed {processed}/{len(image_paths)} images")
        
        return results
    
    def predict_directory(self, directory_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Run inference on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            List of prediction results
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            f for f in directory_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.warning(f"No image files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        return self.predict_batch(image_paths)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Union[str, Path]):
        """
        Save prediction results to file.
        
        Args:
            results: List of prediction results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        
        # Determine format based on extension
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_path.suffix.lower() == '.csv':
            import pandas as pd
            
            # Flatten results for CSV
            flat_results = []
            for result in results:
                if 'error' in result:
                    continue
                
                flat_result = {
                    'image_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'inference_time_ms': result['inference_time_ms'],
                    'timestamp': result['timestamp']
                }
                
                # Add individual probabilities
                for class_name, prob in result['probabilities'].items():
                    flat_result[f'prob_{class_name}'] = prob
                
                flat_results.append(flat_result)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
        else:
            # Default to JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of prediction results."""
        if not results:
            print("No results to summarize")
            return
        
        # Count predictions by class
        class_counts = {}
        total_confidence = 0
        total_inference_time = 0
        error_count = 0
        
        for result in results:
            if 'error' in result:
                error_count += 1
                continue
            
            predicted_class = result['predicted_class']
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            total_confidence += result['confidence']
            total_inference_time += result['inference_time_ms']
        
        valid_results = len(results) - error_count
        
        print(f"\n Prediction Summary")
        print(f"{'='*50}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {valid_results}")
        print(f"Errors: {error_count}")
        
        if valid_results > 0:
            print(f"Average confidence: {total_confidence/valid_results:.3f}")
            print(f"Average inference time: {total_inference_time/valid_results:.2f} ms")
            
            print(f"\n Detection Results:")
            for class_name in self.class_names:
                count = class_counts.get(class_name, 0)
                percentage = (count / valid_results) * 100 if valid_results > 0 else 0
                print(f"  {class_name.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='PCB Defect Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python -m api.inference --image pcb_sample.jpg
  
  # Batch processing
  python -m api.inference --batch ./images/ --output results.json
  
  # Custom model and device
  python -m api.inference --image sample.jpg --model best_model.pth --device cuda
  
  # Save results as CSV
  python -m api.inference --batch ./images/ --output results.csv --format csv
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='Path to single image file')
    input_group.add_argument('--batch', type=str, help='Path to directory with images')
    
    # Model and device arguments
    parser.add_argument('--model', type=str, help='Path to model weights file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps', 'auto'], 
                       default='auto', help='Device to use for inference')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for processing multiple images')
    
    # Output arguments
    parser.add_argument('--output', type=str, help='Path to save results file')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Suppress output except errors')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    try:
        # Initialize inference engine
        inference = PCBDefectInference(
            model_path=args.model,
            device=args.device,
            batch_size=args.batch_size
        )
        
        # Run inference
        if args.image:
            # Single image
            result = inference.predict_single(args.image)
            results = [result]
            
            if not args.quiet:
                print(f"\n Prediction for {args.image}:")
                if 'error' in result:
                    print(f" Error: {result['error']}")
                else:
                    print(f" Predicted class: {result['predicted_class']}")
                    print(f" Confidence: {result['confidence']:.3f}")
                    print(f" Description: {result['description']}")
                    print(f"⏱ Inference time: {result['inference_time_ms']:.2f} ms")
                    
                    print(f"\n All class probabilities:")
                    for class_name, prob in sorted(result['probabilities'].items(), 
                                                 key=lambda x: x[1], reverse=True):
                        print(f"  {class_name.replace('_', ' ').title()}: {prob:.3f}")
        
        else:
            # Batch processing
            results = inference.predict_directory(args.batch)
            
            if not args.quiet:
                inference.print_summary(results)
        
        # Save results if output path specified
        if args.output:
            # Ensure correct extension for format
            output_path = Path(args.output)
            if args.format == 'csv' and output_path.suffix.lower() != '.csv':
                output_path = output_path.with_suffix('.csv')
            elif args.format == 'json' and output_path.suffix.lower() != '.json':
                output_path = output_path.with_suffix('.json')
            
            inference.save_results(results, output_path)
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
