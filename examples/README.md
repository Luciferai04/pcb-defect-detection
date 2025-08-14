#  Usage Examples

This directory contains practical examples demonstrating how to use the PCB defect detection system.

##  Quick Start Examples

### 1. Basic Inference
Process a single PCB image and get predictions.

```bash
# Basic usage
python examples/basic_inference.py --image path/to/pcb.jpg

# With visualization and results saving
python examples/basic_inference.py \
    --image path/to/pcb.jpg \
    --visualize \
    --save-results \
    --output-dir results/
```

**Output:**
- Prediction results in console
- Optional: Visualization image
- Optional: JSON results file

### 2. Batch Processing
Process multiple images in a directory.

```bash
# Process all images in directory
python examples/batch_processing.py \
    --input-dir data/test_images/ \
    --output-dir batch_results/

# With custom model
python examples/batch_processing.py \
    --input-dir data/test_images/ \
    --model checkpoints/best_model.pth \
    --output-dir results/
```

**Output:**
- CSV file with all predictions
- JSON file with detailed results
- Summary report with statistics

##  Example Files

| File | Description | Usage |
|------|-------------|-------|
| `basic_inference.py` | Single image processing | Quick testing and demos |
| `batch_processing.py` | Directory batch processing | Production workflows |
| `README.md` | This file | Documentation |

##  Common Usage Patterns

### Testing New Models
```bash
# Test a newly trained model
python examples/basic_inference.py \
    --image examples/test_pcb.jpg \
    --model checkpoints/latest_model.pth \
    --visualize
```

### Production Batch Processing
```bash
# Process production images
python examples/batch_processing.py \
    --input-dir /path/to/production/images/ \
    --output-dir /path/to/results/ \
    --format both
```

### Quality Assurance Testing
```bash
# Test on validation dataset
python examples/batch_processing.py \
    --input-dir data/validation/ \
    --model checkpoints/best_model.pth \
    --format csv
```

##  Output Formats

### JSON Results
```json
{
  "image_path": "path/to/image.jpg",
  "prediction": {
    "class": "short_circuit",
    "confidence": 0.923,
    "probabilities": {
      "normal": 0.012,
      "short_circuit": 0.923,
      "solder_bridge": 0.043,
      "missing_component": 0.022
    }
  },
  "processing_time": 0.015
}
```

### CSV Results
| filename | prediction | confidence | inference_time |
|----------|------------|------------|----------------|
| pcb1.jpg | short_circuit | 0.923 | 0.015 |
| pcb2.jpg | normal | 0.891 | 0.012 |

##  Customization

### Adding New Examples
Create new example scripts following this pattern:

```python
#!/usr/bin/env python3
"""
Description of your example.
"""

import sys
sys.path.append('.')

from core.foundation_adapter import FoundationAdapter

def main():
    # Your example code here
    pass

if __name__ == "__main__":
    exit(main())
```

### Modifying Existing Examples
All examples are designed to be easily customizable:

1. **Model parameters**: Change backbone, LoRA settings
2. **Input/output paths**: Customize file locations
3. **Processing options**: Add new visualization or analysis
4. **Export formats**: Add new output formats

##  Troubleshooting

### Common Issues

**ImportError: No module named 'core'**
```bash
# Make sure you're running from the project root
cd /path/to/pcb-defect-detection
python examples/basic_inference.py --help
```

**FileNotFoundError: Model checkpoint not found**
```bash
# Train a model first or use randomly initialized weights
python examples/basic_inference.py --image test.jpg  # Uses random weights
```

**CUDA out of memory**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python examples/basic_inference.py --image test.jpg
```

### Getting Help

1. Check the main README.md for setup instructions
2. Run examples with `--help` flag for usage information
3. Check the troubleshooting section in the main README
4. Open an issue on GitHub if problems persist

##  Next Steps

After trying these examples:

1. **Train your own model**: See training documentation
2. **Deploy the API**: Use the FastAPI server for production
3. **Integrate into workflows**: Adapt examples for your specific use case
4. **Extend functionality**: Add custom preprocessing or post-processing

For more advanced usage, see the main documentation and API reference.
