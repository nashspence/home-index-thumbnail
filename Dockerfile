FROM python:slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies for home-index-thumbnail
RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    ffmpeg \
    webp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# run imagemagick easy installer
RUN t=$(mktemp) && \
    wget 'https://dist.1-2.dev/imei.sh' -qO "$t" && \
    bash "$t" && \
    rm "$t"

RUN identify -version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/home_index_thumbnail .

ENTRYPOINT ["python3", "/app/main.py"]
