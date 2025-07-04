# E-commerce Facecare Recommender 

A Content-based recommendation system for facecare products in an e-commerce.

## Project Structure

```
ecom-facecare-recommender/
│
├── src/                          # Main source code directory
│   ├── __init__.py
│   ├── main.py                   # Application entry point
│   │
│   ├── core/                     # Core configuration and settings
│   │   ├── __init__.py
│   │   └── config.py            # Application configuration
│   │
│   ├── models/                   # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py           # Pydantic models for API
│   │
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── data_service.py      # Data loading and management
│   │   └── recommendation_service.py  # Recommendation algorithms
│   │
│   ├── api/                      # API routes and endpoints
│   │   ├── __init__.py
│   │   └── routes.py            # FastAPI routes
│   │
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logging.py           # Logging configuration
│       └── exceptions.py        # Custom exceptions
│
├── data/                         # Data files (CSV files)
├── embeddings/                   # Pre-computed embeddings
│   ├── vie_bi_encode_embedding.npy
│   └── vietnamese_embedding.npy
│
├── static/                       # Static web assets
├── templates/                    # HTML templates
├── logs/                         # Application logs
├── notebooks/                    # Jupyter notebooks
│
├── app.py                        # Legacy entry point (backward compatible)
├── main.py                       # New entry point
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
├── Dockerfile                    # Docker configuration
└── README_RESTRUCTURED.md        # This file
```

## Data
You can find the datasets in Kaggle, link: [![Kaggle](https://img.shields.io/badge/Kaggle-Open%20in%20Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/danielway17/face-clean-ecomer)

## Key Improvements

### 1. Separation of Concerns
- **Configuration**: Centralized in `src/core/config.py`
- **Data Access**: Isolated in `src/services/data_service.py`
- **Business Logic**: Separated in `src/services/recommendation_service.py`
- **API Layer**: Clean routes in `src/api/routes.py`
- **Models**: Type-safe schemas in `src/models/schemas.py`

### 2. Error Handling
- Custom exception classes in `src/utils/exceptions.py`
- Centralized exception handling
- Proper logging throughout the application

### 3. Configuration Management
- Environment-based configuration
- Type-safe settings using Pydantic
- Easy to modify for different environments

### 4. Data Loading
- Data is loaded only when needed
- Singleton pattern for services
- Better memory management

### 5. Logging
- Structured logging with Loguru
- Multiple log levels and outputs
- Rotation and retention policies

### 6. Type Safety
- Full type annotations
- Pydantic models for data validation
- Better IDE support and error catching

## Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/ecom-facecare-recommender.git
cd ecom-facecare-recommender
```

### 2. Create and activate conda env 
```bash
conda create -n face_clean_env python=3.9
conda activate face_clean_env
```

### 3. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 4. Running the Application

```bash
python app.py
```

### 5. API Documentation
Once running, visit:
- Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative API Docs: http://localhost:8000/redoc

## Configuration

The application can be configured through environment variables or the `.env` file:

```bash
# Server settings
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Data paths
DATABASE__POPULAR_PRODUCTS_PATH=./data/top_brand_products.csv
DATABASE__COMBINED_PRODUCTS_PATH=./data/data_product_combine.csv
# ...
```

## Adding New Features

### 1. Adding a New API Endpoint
1. Define the route in `src/api/routes.py`
2. Add any new data models to `src/models/schemas.py`

### 2. Adding a New Recommendation Algorithm
1. Add the algorithm to `src/services/recommendation_service.py`
2. Update configuration in `src/core/config.py` if needed
3. Add any new data dependencies to `src/services/data_service.py`

### 3. Adding New Data Sources
1. Extend `src/services/data_service.py` with new loading methods
2. Update configuration paths in `src/core/config.py`
3. Add new data models to `src/models/schemas.py`

## Deployment

### Docker
```bash
# Build image
docker build -t ecom-recommender .

# Run container
docker run -p 8000:8000 ecom-recommender
```

