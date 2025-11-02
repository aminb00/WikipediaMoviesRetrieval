FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt punkt_tab stopwords --quiet

# Copy all project files
COPY . .

# Create data directories
RUN mkdir -p data Data

# Download dataset if CSV files don't exist
# Note: download_dataset.py writes to data/, so we copy to both data/ and Data/
RUN (python download_dataset.py 2>&1 || echo "Download attempted") && \
    (if [ -d data ] && [ ! -z "$(ls data/*.csv 2>/dev/null)" ]; then \
       find data -name "*.csv" -exec cp {} Data/ \; 2>/dev/null || true; \
       echo "CSV files copied to Data/"; \
     fi) && \
    (if [ ! -z "$(ls Data/*.csv 2>/dev/null)" ] || [ ! -z "$(ls data/*.csv 2>/dev/null)" ]; then \
       echo "✓ CSV files ready"; \
     else \
       echo "⚠ WARNING: No CSV files found. Dataset download may require Kaggle API credentials."; \
     fi)

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "test_cli.py"]

