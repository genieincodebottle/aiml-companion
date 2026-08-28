FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# Create artifact directories
RUN mkdir -p artifacts/figures artifacts/results data/raw

# Expose FastAPI port
EXPOSE 8000

# Default: run the full pipeline
CMD ["python", "main.py"]

# To serve predictions instead:
# CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]