FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    build-essential \
    python3-dev \
    libffi-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies first
COPY flashcamp/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Install heavy packages with pre-built wheels
RUN pip install --user --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY flashcamp/ /app/flashcamp/
COPY pyproject.toml .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create necessary directories
RUN mkdir -p /app/tmp /app/logs /app/reports/pdf

EXPOSE 8000
CMD ["uvicorn", "flashcamp.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"] 