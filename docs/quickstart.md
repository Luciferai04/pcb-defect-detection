# Quick Start Guide

Get up and running with PCB defect detection in minutes! This guide walks you through the essential steps from installation to your first inference.

##  5-Minute Setup

### Step 1: Installation

```bash
# Clone and setup
git clone https://github.com/soumyajitghosh/pcb-defect-detection.git
cd pcb-defect-detection
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(' Setup complete!')"
```

### Step 2: Download Sample Data

```bash
# Create sample data directory
mkdir -p data/samples

# Download sample PCB images (replace with actual URLs)
wget -O data/samples/pcb_normal.jpg "https://example.com/sample_pcb_normal.jpg"
wget -O data/samples/pcb_defect.jpg "https://example.com/sample_pcb_defect.jpg"
```

### Step 3: Quick Inference

```bash
# Run inference on sample image
python -m api.inference --image data/samples/pcb_defect.jpg
```

##  Basic Usage Examples

### 1. Load and Use Pre-trained Model

```python
from enhanced_pcb_model import create_enhanced_model
import torch
from PIL import Image
import torchvision.transforms as transforms

# Create model
model, loss_fn = create_enhanced_model(num_classes=5)

# Load pre-trained weights (if available)
# model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('data/samples/pcb_defect.jpg')
image_tensor = transform(image).unsqueeze(0)

model.eval()
with torch.no_grad():
    prediction = model(image_tensor)
    predicted_class = torch.argmax(prediction, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
```

### 2. Foundation Model Adapter

```python
from core.foundation_adapter import create_foundation_adapter

# Create adapter for medical domain
adapter = create_foundation_adapter(
    method="AD-CLIP",
    domain="medical",
    rank=4,
    alpha=32
)

print(f"Parameter efficiency: {adapter.efficiency_ratio:.4f}")
print(f"Trainable parameters: {adapter.trainable_parameters:,}")
```

### 3. Active Learning Pipeline

```python
from train_active_learning import ActiveLearningConfig
import torch

# Configure active learning
config = ActiveLearningConfig()
config.al_strategy = 'hybrid'
config.initial_labeled_size = 100
config.budget_per_round = 50

print(f"Strategy: {config.al_strategy}")
print(f"Device: {config.device}")
```

##  Configuration

### Basic Configuration (`config.py`)

```python
import torch

# Model Configuration
MODEL_CONFIG = {
    'backbone': 'resnet50',
    'num_classes': 5,
    'lora_rank': 4,
    'lora_alpha': 32,
    'dropout_rate': 0.3,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'num_epochs': 100,
    'device': 'auto',  # auto, cpu, cuda, mps
}

# Data Configuration
DATA_CONFIG = {
    'image_size': 224,
    'data_dir': './data/pcb_defects',
    'num_workers': 4,
    'pin_memory': True,
}

# Class names for PCB defects
CLASS_NAMES = [
    'normal',
    'missing_component', 
    'solder_bridge',
    'misalignment',
    'short_circuit'
]
```

##  Training Your First Model

### Quick Training Script

```python
#!/usr/bin/env python3
"""Quick training example"""

import torch
from enhanced_pcb_model import create_enhanced_model
from torch.utils.data import DataLoader, TensorDataset

# Create synthetic data for demo
def create_demo_dataset(num_samples=1000):
    """Create synthetic PCB data for demonstration"""
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 5, (num_samples,))
    return TensorDataset(images, labels)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model and data
model, loss_fn = create_enhanced_model(num_classes=5)
model.to(device)

train_dataset = create_demo_dataset(800)
val_dataset = create_demo_dataset(200)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Optimizer
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], 
    lr=1e-3, 
    weight_decay=1e-4
)

# Training loop
model.train()
for epoch in range(5):  # Quick demo with 5 epochs
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch} average loss: {total_loss/len(train_loader):.4f}")

print(" Training completed!")
```

##  API Server

### Start the API Server

```bash
# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or with gunicorn for production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Basic info
curl http://localhost:8000/

# Upload image for prediction (when implemented)
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/samples/pcb_defect.jpg"
```

##  Docker Deployment

### Build and Run

```bash
# Build image
docker build -t pcb-defect-detection:latest .

# Run container
docker run -d -p 8000:8000 --name pcb-api pcb-defect-detection:latest

# Check logs
docker logs pcb-api

# Stop container
docker stop pcb-api
```

### Docker Compose (Advanced)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cpu
      - LOG_LEVEL=INFO
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    restart: unless-stopped
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m "unit"           # Unit tests only
pytest -m "integration"    # Integration tests
pytest -m "not slow"       # Skip slow tests

# Run tests in parallel
pytest -n auto
```

##  Monitoring & Logging

### Weights & Biases Integration

```python
import wandb

# Initialize W&B
wandb.init(
    project="pcb-defect-detection",
    config={
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "architecture": "ResNet50+LoRA",
    }
)

# Log metrics during training
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "val_accuracy": val_accuracy,
    "parameter_efficiency": efficiency_ratio
})
```

### Basic Logging Setup

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started")
```

##  Next Steps

Congratulations! You now have a working PCB defect detection system. Here's what to explore next:

### Immediate Next Steps
1. **[Follow detailed tutorials](tutorials/index.md)** - Learn advanced features
2. **[Read the user guide](user_guide/foundation_models.md)** - Understand the concepts
3. **[Explore the API reference](api/core.md)** - Dive into implementation details

### Advanced Topics
1. **[Hyperparameter Optimization](user_guide/model_training.md#hyperparameter-optimization)**
2. **[Production Deployment](deployment/production.md)**
3. **[Custom Domain Adaptation](research/methodology.md)**
4. **[Research Methodology](research/experiments.md)**

### Get Involved
1. **[Contributing Guide](contributing.md)** - Help improve the project
2. **[GitHub Issues](https://github.com/soumyajitghosh/pcb-defect-detection/issues)** - Report bugs or request features
3. **[Discussions](https://github.com/soumyajitghosh/pcb-defect-detection/discussions)** - Ask questions and share ideas

##  Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce batch size: `config.batch_size = 16` |
| **Import errors** | Check installation: `pip install -r requirements.txt` |
| **Slow training** | Use GPU: `config.device = 'cuda'` |
| **Poor accuracy** | Increase epochs: `config.num_epochs = 200` |

### Getting Help

- **Documentation**: You're reading it! 
- **Issues**: [GitHub Issues](https://github.com/soumyajitghosh/pcb-defect-detection/issues) 
- **Discussions**: [GitHub Discussions](https://github.com/soumyajitghosh/pcb-defect-detection/discussions) 
- **Email**: research@example.com 

Ready to detect some defects? Let's go! 
