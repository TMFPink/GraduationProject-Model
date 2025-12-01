# Stage 1: Model Download
FROM python:3.10.12-slim as model-downloader

WORKDIR /tmp

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Install only necessary packages for model download
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch torchvision open_clip_torch

# Download model with memory constraints
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_HOME=/tmp/models

RUN python -c "import torch"
RUN python -c "torch.set_num_threads(1)"
RUN python -c "import open_clip"
RUN python -c "print('Downloading CLIP model...')"
RUN python -c "model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')"
RUN python -c "print('Model downloaded successfully')"

# Stage 2: Final Image
FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy downloaded models from first stage
COPY --from=model-downloader /tmp/models /home/app/.cache

RUN groupadd --system app && useradd --system --gid app app

COPY . .
RUN mkdir -p /app/models /home/app/.config/Ultralytics && \
    chown -R app:app /app /home/app

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/card_model_29_10.pt
ENV TORCH_HOME=/home/app/.cache

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]