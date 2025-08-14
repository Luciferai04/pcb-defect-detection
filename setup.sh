#!/bin/bash
# PCB Defect Detection - Quick Setup Script
# This script automates the installation process for new users

set -e

echo "🔍 PCB Defect Detection - Quick Setup"
echo "===================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Python is installed and version is correct
check_python() {
    print_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.10 or higher."
        echo "Visit: https://www.python.org/downloads/"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d" " -f2)
    major_version=$(echo $python_version | cut -d"." -f1)
    minor_version=$(echo $python_version | cut -d"." -f2)
    
    if [[ $major_version -lt 3 ]] || [[ $major_version -eq 3 && $minor_version -lt 10 ]]; then
        print_error "Python version $python_version is not supported. Please upgrade to Python 3.10+"
        exit 1
    fi
    
    print_status "Python $python_version is compatible"
}

# Check if Git is installed
check_git() {
    print_info "Checking Git installation..."
    
    if ! command -v git &> /dev/null; then
        print_warning "Git is not installed. Installing Git..."
        
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install git
            else
                print_error "Please install Homebrew first: https://brew.sh/"
                exit 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            sudo apt-get update && sudo apt-get install -y git
        else
            print_error "Please install Git manually: https://git-scm.com/downloads"
            exit 1
        fi
    fi
    
    print_status "Git is installed"
}

# Create virtual environment
setup_venv() {
    print_info "Setting up virtual environment..."
    
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    print_status "Virtual environment created and activated"
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies..."
    
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found. Are you in the correct directory?"
        exit 1
    fi
    
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully"
}

# Run tests to verify installation
run_tests() {
    print_info "Running tests to verify installation..."
    
    if command -v pytest &> /dev/null; then
        if [[ -d "tests" ]]; then
            pytest tests/ -v --tb=short
            print_status "All tests passed"
        else
            print_warning "Tests directory not found, skipping tests"
        fi
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Download sample data (optional)
setup_sample_data() {
    print_info "Setting up sample data..."
    
    mkdir -p data/samples
    mkdir -p examples
    
    # Create placeholder files
    echo "# Sample PCB images will be placed here" > data/samples/README.md
    echo "# Example scripts and notebooks" > examples/README.md
    
    print_status "Sample data directories created"
}

# Final setup verification
verify_setup() {
    print_info "Verifying installation..."
    
    # Test import
    python3 -c "
import sys
sys.path.append('.')
try:
    from core.foundation_adapter import FoundationAdapter
    print('✓ Core modules can be imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠ Warning: {e}')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__} is available')
    if torch.cuda.is_available():
        print(f'✓ CUDA is available (GPU: {torch.cuda.get_device_name(0)})')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('✓ Apple MPS is available')
    else:
        print('ℹ Running on CPU (GPU acceleration not available)')
except Exception as e:
    print(f'⚠ PyTorch issue: {e}')
"
}

# Main setup function
main() {
    echo "Starting automated setup..."
    echo ""
    
    # Check system requirements
    check_python
    check_git
    
    # Setup Python environment
    setup_venv
    install_deps
    
    # Setup project structure
    setup_sample_data
    
    # Verify installation
    verify_setup
    run_tests
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the demo: python demo.py"
    echo "3. Start the API server: python -m api.main"
    echo "4. Check the documentation: open docs/quickstart.md"
    echo ""
    echo "🔗 Useful commands:"
    echo "- make test          # Run all tests"
    echo "- make lint          # Check code quality"
    echo "- make docs          # Generate documentation"
    echo "- make api           # Start API server"
    echo ""
    echo "📖 For detailed usage, see: README.md"
    echo "🐛 For issues, visit: https://github.com/your-username/pcb-defect-detection/issues"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "PCB Defect Detection Setup Script"
        echo ""
        echo "Usage: ./setup.sh [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --skip-tests   Skip running tests after installation"
        echo "  --gpu-only     Install GPU-specific dependencies only"
        echo ""
        echo "This script will:"
        echo "1. Check Python and Git installation"
        echo "2. Create virtual environment"
        echo "3. Install dependencies"
        echo "4. Setup project structure"
        echo "5. Verify installation"
        echo "6. Run tests (optional)"
        ;;
    --skip-tests)
        export SKIP_TESTS=1
        main
        ;;
    --gpu-only)
        print_info "Installing GPU-specific dependencies..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    *)
        main
        ;;
esac
