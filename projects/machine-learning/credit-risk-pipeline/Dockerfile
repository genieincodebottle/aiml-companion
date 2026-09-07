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
# run.py is the documented entry point everywhere else, so the image must have
# it too. Without this line `docker run ... python run.py` fails with a bare
# "can't open file", which is a confusing first experience of the container.
COPY run.py .
# The German Credit CSV ships with the repo. Without it the container has to
# reach OpenML on first run, so `docker run` fails on an air-gapped or
# offline machine for no reason the message explains.
COPY data/ ./data/

# Create artifact directories
RUN mkdir -p artifacts/figures artifacts/results data/raw

# Expose FastAPI port
EXPOSE 8000

# Default: run the full pipeline
CMD ["python", "run.py"]

# To serve predictions instead:
# CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]