# Build from the repo root of this project:
#   docker build -f k8s/Dockerfile -t engine-sim:dev .
#   kind load docker-image engine-sim:dev
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi~=0.115" "uvicorn~=0.32"
COPY engine_sim /app/engine_sim
EXPOSE 8100
ENTRYPOINT ["python", "-m", "engine_sim", "--host", "0.0.0.0", "--port", "8100"]
CMD ["--gpu", "l4", "--quant", "bf16", "--time-scale", "10"]
