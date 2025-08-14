# Installation Guide

This guide provides comprehensive installation instructions for the PCB Defect Detection Research framework.

## Requirements

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB free disk space

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8GB | 32GB+ |
| **GPU** | GTX 1060 | RTX 3080+ |
| **Storage** | 10GB | 100GB+ SSD |

## Installation Methods

### Method 1: Pip Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/soumyajitghosh/pcb-defect-detection.git
cd pcb-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Method 2: Conda Installation

```bash
# Clone repository
git clone https://github.com/soumyajitghosh/pcb-defect-detection.git
cd pcb-defect-detection

# Create conda environment
conda create -n pcb-detection python=3.9
conda activate pcb-detection

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation

```bash
# Pull the Docker image
docker pull pcb-defect-detection:latest

# Or build from source
git clone https://github.com/soumyajitghosh/pcb-defect-detection.git
cd pcb-defect-detection
docker build -t pcb-defect-detection:latest .

# Run the container
docker run -p 8000:8000 pcb-defect-detection:latest
```

## Development Installation

For development and contribution:

```bash
# Clone with development setup
git clone https://github.com/soumyajitghosh/pcb-defect-detection.git
cd pcb-defect-detection

# Install with development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## GPU Setup

### CUDA Installation

1. **Check CUDA compatibility**:
   ```bash
   nvidia-smi
   ```

2. **Install CUDA toolkit** (if needed):
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
   sudo apt-key add /var/cuda-repo-ubuntu2004-11-8-local/7fa2af80.pub
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

3. **Verify CUDA setup**:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Apple Silicon (MPS) Setup

For Apple Silicon Macs:

```bash
# Verify MPS availability
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Model configurations
MODEL_PATH=./checkpoints/best_model.pth
DEVICE=auto  # auto, cpu, cuda, mps

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=pcb-defect-detection
```

### Configuration Files

1. **Model Configuration** (`config.py`):
   ```python
   MODEL_CONFIG = {
       'backbone': 'resnet50',
       'num_classes': 5,
       'lora_rank': 4,
       'lora_alpha': 32,
   }
   ```

2. **Training Configuration**:
   ```python
   TRAINING_CONFIG = {
       'batch_size': 32,
       'learning_rate': 1e-3,
       'num_epochs': 100,
       'device': 'auto',
   }
   ```

## Verification

### Quick Test

```bash
# Test core functionality
python -c "
from core.foundation_adapter import create_foundation_adapter
adapter = create_foundation_adapter('AD-CLIP', 'medical')
print(' Foundation adapter created successfully')
"

# Test enhanced model
python -c "
from enhanced_pcb_model import create_enhanced_model
model, loss_fn = create_enhanced_model()
print(' Enhanced PCB model created successfully')
"
```

### Run Test Suite

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "unit"      # Run unit tests only
```

### API Test

```bash
# Start the API server
python -m uvicorn main:app --reload

# Test endpoints (in another terminal)
curl http://localhost:8000/health
curl http://localhost:8000/
```

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Solution: Reduce batch size
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 2. **Import Errors**
```bash
# Solution: Install missing dependencies
pip install --upgrade -r requirements.txt
```

#### 3. **Permission Errors**
```bash
# Solution: Fix permissions
sudo chown -R $USER:$USER ./checkpoints/
sudo chown -R $USER:$USER ./outputs/
```

#### 4. **Docker Issues**
```bash
# Solution: Clean Docker cache
docker system prune -a
docker build --no-cache -t pcb-defect-detection:latest .
```

### Performance Optimization

1. **Memory Optimization**:
   ```bash
   # Use gradient checkpointing
   export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6
   ```

2. **Speed Optimization**:
   ```bash
   # Enable optimized attention
   pip install flash-attn
   ```

3. **Multi-GPU Setup**:
   ```python
   # In your training script
   model = torch.nn.DataParallel(model)
   ```

## Next Steps

After installation:

1. **Read the [Quick Start Guide](quickstart.md)**
2. **Follow the [Tutorials](tutorials/index.md)**
3. **Explore the [API Reference](api/core.md)**
4. **Check out [Research Results](research/results.md)**

## Getting Help

- **Documentation**: [Full Documentation](index.rst)
- **Issues**: [GitHub Issues](https://github.com/soumyajitghosh/pcb-defect-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/soumyajitghosh/pcb-defect-detection/discussions)
- **Email**: research@example.com
