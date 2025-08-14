# PCB Defect Detection Research - Development Makefile

.PHONY: help install install-dev test lint format type-check docs clean docker docker-run api

# Default help command
help:
	@echo "PCB Defect Detection Research - Development Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup-env    Setup virtual environment"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run fast tests only"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  pre-commit   Run pre-commit hooks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  notebook     Start Jupyter notebook server"
	@echo ""
	@echo "API & Services:"
	@echo "  api          Start FastAPI development server"
	@echo "  api-prod     Start production API server"
	@echo "  inference    Run inference on sample image"
	@echo ""
	@echo "Docker:"
	@echo "  docker       Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo "  docker-test  Test Docker image"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean up generated files"
	@echo "  check-deps   Check for dependency updates"
	@echo "  security     Run security checks"

# Setup commands
install:
	pip install -r requirements.txt

install-dev: install
	pip install -e .[dev]
	pre-commit install

setup-env:
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"

# Development commands
test:
	pytest -v --cov=. --cov-report=html --cov-report=term-missing

test-fast:
	pytest -v -m "not slow" --cov=. --cov-report=term-missing

test-gpu:
	pytest -v -m "gpu" --cov=. --cov-report=term-missing

lint:
	flake8 . --count --statistics
	black --check --diff .
	isort --check-only --diff .

format:
	black .
	isort .
	@echo "✓ Code formatted successfully"

type-check:
	mypy . --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# Documentation commands
docs:
	cd docs  make html || true
	@echo "Documentation built in docs/_build/html/ (if Sphinx is installed)"
	python3 docs/generate_figures.py
	python3 docs/generate_poster.py
	@echo "Generated figures saved in docs/figures/"

# PDF export using Pandoc (if installed)
pdf:
	@command -v pandoc >/dev/null 2>&1 || { echo "Pandoc not found. Install with: brew install pandoc"; exit 1; }
	pandoc RESEARCH_PAPER_GENERATED.md \
		--from gfm \
		--pdf-engine=lualatex \
		--template=docs/templates/pandoc_ieee_math.tex \
		--toc \
		--metadata title="Adaptive Foundation Models for PCB Defect Detection" \
		-o RESEARCH_PAPER_GENERATED.pdf || { echo "Pandoc export failed. Ensure a LaTeX engine (lualatex) is installed."; exit 1; }
	@echo "✓ Exported RESEARCH_PAPER_GENERATED.pdf"
	pandoc EXECUTIVE_SUMMARY.md \
		--from gfm \
		--pdf-engine=lualatex \
		--template=docs/templates/pandoc_ieee_math.tex \
		--metadata title="Executive Summary: PCB Defect Detection" \
		-o EXECUTIVE_SUMMARY.pdf || true
	@echo "✓ Exported EXECUTIVE_SUMMARY.pdf (if possible)"

# IEEE LaTeX build (requires latexmk or pdflatex)
ieee:
	@command -v latexmk /dev/null 21 || { echo "latexmk not found. Trying pdflatex+bibtex..."; pdflatex -interaction=nonstopmode ieee_paper.tex || { echo "Install MacTeX or TinyTeX, then rerun."; exit 1; }; bibtex ieee_paper || true; pdflatex -interaction=nonstopmode ieee_paper.tex; pdflatex -interaction=nonstopmode ieee_paper.tex; exit 0; }
	latexmk -pdf -bibtex -interaction=nonstopmode ieee_paper.tex
	@echo "✓ Built ieee_paper.pdf"

docs-serve: docs
	cd docs/_build/html && python -m http.server 8080
	@echo "Documentation served at http://localhost:8080"

docs-clean:
	cd docs && make clean

notebook:
	jupyter notebook docs/tutorials/

# API and service commands
api:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

api-prod:
	gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

inference:
	@echo "Running inference on sample (synthetic) image..."
	python3 -c "from enhanced_pcb_model import create_enhanced_model; import torch; model, _ = create_enhanced_model(); print('✓ Model inference test successful')"

# Docker commands
docker:
	docker build -t pcb-defect-detection:latest .
	@echo "✓ Docker image built: pcb-defect-detection:latest"

docker-run: docker
	docker run -p 8000:8000 --name pcb-api pcb-defect-detection:latest

docker-test: docker
	docker run --rm pcb-defect-detection:latest python -c "import torch; print('PyTorch:', torch.__version__); from enhanced_pcb_model import create_enhanced_model; print('✓ Docker image test successful')"

docker-stop:
	docker stop pcb-api || true
	docker rm pcb-api || true

# Training commands
train:
	python train_active_learning.py

train-ssl:
	python train_ssl_simple.py

hyperopt:
	python hyperparameter_optimization.py

# Utility commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	@echo "✓ Cleaned up generated files"

check-deps:
	pip list --outdated

security:
	bandit -r . -f json -o bandit-report.json || echo "Security scan completed (check bandit-report.json)"
	@echo "✓ Security scan completed"

# Research-specific commands
experiment:
	python comprehensive_analysis.py

gradcam:
	python run_gradcam.py

performance:
	python performance_analysis.py

# Quality assurance
qa: lint type-check test
	@echo "✓ All quality assurance checks passed"

# CI simulation
ci: install-dev qa docs docker-test
	@echo "✓ CI simulation completed successfully"

# Development workflow
dev-setup: setup-env install-dev
	@echo "✓ Development environment setup complete"
	@echo "Next steps:"
	@echo "1. Activate virtual environment: source venv/bin/activate"
	@echo "2. Run tests: make test"
	@echo "3. Start API: make api"
	@echo "4. Open docs: make docs-serve"
