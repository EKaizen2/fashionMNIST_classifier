# Makefile for Fashion MNIST Classifier

# Python interpreter
PYTHON = python3

# Directories
LOGS_DIR = logs
DATA_DIR = .

# Main script
CLASSIFIER = classifier.py

# Default target
.PHONY: all
all: setup run

# Setup: Create necessary directories and download dataset if needed
.PHONY: setup
setup:
	@echo "Setting up project..."
	@mkdir -p $(LOGS_DIR)
	@$(PYTHON) -c "from torchvision import datasets; \
		datasets.FashionMNIST('$(DATA_DIR)', download=True)"

# Run the classifier
.PHONY: run
run:
	@echo "Running Fashion MNIST classifier..."
	@$(PYTHON) $(CLASSIFIER)

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	@rm -f $(LOGS_DIR)/*.txt
	@rm -rf FashionMNIST
	@rm -rf __pycache__
	@rm -f *.pyc

# Clean logs only
.PHONY: clean-logs
clean-logs:
	@echo "Cleaning log files..."
	@rm -f $(LOGS_DIR)/*.txt

# Show logs
.PHONY: logs
logs:
	@if [ -f $(LOGS_DIR)/logs.txt ]; then \
		cat $(LOGS_DIR)/logs.txt; \
	else \
		echo "No logs found."; \
	fi

# Install dependencies
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	@$(PYTHON) -m pip install torch torchvision pillow numpy matplotlib

# Help target
.PHONY: help
help:
	@echo "Fashion MNIST Classifier Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Setup and run the classifier (default)"
	@echo "  setup      - Create directories and download dataset"
	@echo "  run        - Run the classifier"
	@echo "  clean      - Remove all generated files"
	@echo "  clean-logs - Remove only log files"
	@echo "  logs       - Display logs"
	@echo "  deps       - Install Python dependencies"
	@echo "  help       - Show this help message" 