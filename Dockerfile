FROM python:3.12-slim

WORKDIR /app

# Install PyTorch CPU (smaller image, no CUDA overhead)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers

# Copy source
COPY . .

# Pre-download the granite model at build time so containers start instantly
RUN python -c "from thalamus.granite_embedder import get_embedder; get_embedder(); print('Granite cached')"

# Default: run the interactive REPL
CMD ["python", "run.py"]
