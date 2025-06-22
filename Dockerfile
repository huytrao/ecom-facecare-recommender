FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Create necessary directories
RUN mkdir -p data embeddings models static/image templates

# Use volume mounts for large directories instead of copying
VOLUME ["/app/embeddings", "/app/models", "/app/static/image"]

# Copy only essential data files
COPY data/ ./data/
COPY templates/ ./templates/
COPY static/style.css ./static/

# Ensure correct port configuration
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# docker run -p 8080:8080 \
#   -v /path/to/local/embeddings:/app/embeddings \
#   -v /path/to/local/models:/app/models \
#   -v /path/to/local/static/image:/app/static/image \
#   recommendation-app
# docker build -t recommendation-app .