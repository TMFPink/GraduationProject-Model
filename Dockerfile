FROM python:3.10.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd --system app && useradd --system --gid app app

COPY . .
RUN mkdir -p /app/models /home/app/.config/Ultralytics && chown -R app:app /app /home/app

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/card_model_29_10.pt

USER app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]