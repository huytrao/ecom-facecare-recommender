.PHONY: help install dev run test clean format lint

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	@echo "Create conda environment and install dependencies"
	@echo "Creating conda environment 'face_clean_env' with Python 3.9"
	conda create -n face_clean_env python=3.9
	conda activate face_clean_env
	@echo "Installing dependencies..."
	pip install -r requirements.txt

dev:  ## Run in development mode
	python dev.py

run:  ## Run the application (legacy)
	python app.py

test:  ## Run tests (placeholder)
	@echo "Tests not implemented yet"

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete

format:  ## Format code with black
	black src/ --line-length 88

lint:  ## Lint code (placeholder)
	@echo "Linting not configured yet"

health:  ## Check application health
	curl -s http://localhost:8000/health | python -m json.tool

logs:  ## View application logs
	tail -f logs/app.log

docker-build:  ## Build Docker image
	docker build -t ecom-recommender .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 ecom-recommender
