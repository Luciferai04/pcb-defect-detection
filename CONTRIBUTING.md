# Contributing to PCB Defect Detection

We welcome contributions to this project! This document provides guidelines for contributing.

##  Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- Basic understanding of PyTorch and computer vision

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/your-username/pcb-defect-detection.git
cd pcb-defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

##  Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Write clean, documented code
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Include docstrings for new functions/classes

### 3. Add Tests
```bash
# Create tests for new functionality
pytest tests/test_your_feature.py

# Run all tests
pytest tests/
```

### 4. Commit Changes
```bash
# Use conventional commit messages
git commit -m "feat: add new feature description"
git commit -m "fix: resolve bug in module"
git commit -m "docs: update README"
```

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
# Then create a Pull Request on GitHub
```

##  Code Style

### Python Standards
- Follow PEP 8
- Use type hints
- Maximum line length: 88 characters (Black default)
- Use meaningful variable and function names

### Example Code Style
```python
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

class ExampleClass:
    """Example class with proper documentation."""
    
    def __init__(self, param: int) -> None:
        """Initialize with parameter.
        
        Args:
            param: Integer parameter for initialization.
        """
        self.param = param
    
    def process_data(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process input data and return results.
        
        Args:
            data: Input tensor of shape (batch_size, features).
            
        Returns:
            Dictionary containing processed results.
            
        Raises:
            ValueError: If input data has wrong shape.
        """
        if data.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {data.dim()}D")
            
        return {"processed": data * self.param}
```

## 🧪 Testing Guidelines

### Writing Tests
```python
import pytest
import torch
from your_module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.instance = YourClass(param=5)
    
    def test_initialization(self):
        """Test proper initialization."""
        assert self.instance.param == 5
    
    def test_process_data_valid_input(self):
        """Test data processing with valid input."""
        data = torch.randn(10, 32)
        result = self.instance.process_data(data)
        
        assert "processed" in result
        assert result["processed"].shape == data.shape
    
    def test_process_data_invalid_input(self):
        """Test data processing with invalid input."""
        data = torch.randn(10)  # 1D tensor
        
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            self.instance.process_data(data)
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_foundation_adapter.py

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test
pytest tests/test_foundation_adapter.py::TestFoundationAdapter::test_forward
```

##  Documentation

### Code Documentation
- Use docstrings for all public functions, classes, and modules
- Follow Google/NumPy docstring style
- Include examples in docstrings when helpful

### API Documentation
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3
) -> Dict[str, List[float]]:
    """Train a model with given parameters.
    
    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs. Defaults to 10.
        lr: Learning rate. Defaults to 1e-3.
        
    Returns:
        Dictionary containing training history with keys:
        - 'loss': List of loss values per epoch
        - 'accuracy': List of accuracy values per epoch
        
    Example:
        >>> model = MyModel()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> history = train_model(model, loader, epochs=5)
        >>> print(history['loss'])
        [0.8, 0.6, 0.4, 0.3, 0.2]
    """
```

##  Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code to reproduce the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, PyTorch version, OS
6. **Error messages**: Full error traceback if applicable

### Bug Report Template
```markdown
**Bug Description**
Brief description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- Python version: 3.10.0
- PyTorch version: 2.0.0
- OS: Ubuntu 20.04

**Error Message**
```
Full error traceback here
```
```

##  Feature Requests

When suggesting features:

1. **Use case**: Explain why this feature would be useful
2. **Description**: Detailed description of the proposed feature
3. **Implementation ideas**: Rough ideas about implementation
4. **Alternatives**: Alternative solutions you've considered

##  Pull Request Guidelines

### Before Submitting
- [ ] Tests pass (`pytest`)
- [ ] Code is properly formatted (`black`, `isort`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG is updated (if applicable)

### PR Description Template
```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing functionality)
- [ ] Documentation update

## Testing
- [ ] New tests added
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

##  Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## 🤝 Community

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

### Getting Help
- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Be specific about your problem
- Provide minimal reproducible examples

##  Contact

For questions about contributing:
- Open an issue for bugs/features
- Use GitHub Discussions for questions
- Email: research@example.com

Thank you for contributing! 
