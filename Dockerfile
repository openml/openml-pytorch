FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/workspace:${PYTHONPATH}" \
    OPENML_CONFIG_DIR="/root/.openml"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy the rest of the project
COPY . .

# Install in development mode
RUN pip install --no-cache-dir -e .

RUN useradd -m appuser && chown -R appuser:appuser /workspace
USER appuser

CMD ["bash"]
