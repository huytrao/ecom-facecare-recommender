# Ecom Facecare Recommender

A Content-based recommendation system for facecare products in an e-commerce setting.

## Features
---

- Personalized facecare product recommendations
- User-friendly interface
- Scalable and modular codebase
- fix cold start problem

## Directory Structure
---

```
├── app.py
├── data
│   ├── data_product_combine.csv
│   ├── data_product.csv
│   ├── data_purchase.csv
│   ├── data_reviews_purchase.csv
│   └── top_brand_products.csv
├── Dockerfile
├── embeddings
│   ├── vie_bi_encode_embedding.npy
│   └── vietnamese_embedding.npy
├── images
│   └── image.txt
├── ingestion_elastic_search.py
├── notebooks
│   ├── 1_preprocess_data.ipynb
│   ├── 2_embedding_content.ipynb
│   └── 3_content_based_with_popular_score.ipynb
├── README.md
├── requirements.txt
├── static
│   ├── image
│   └── style.css
└── templates
    └── index.html
```

## Getting Started
---

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ecom-facecare-recommender.git
    cd ecom-facecare-recommender
    ```

2. Create and activate conda env 
    ```bash
    conda create -n face_clean_env python=3.9
    conda activate face_clean_env
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run FastAPI server:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8080
    ```