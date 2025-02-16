FROM python:slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive

ENV OPENH264_VERSION=2.5.0

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Download and install Cisco's precompiled libopenh264
RUN wget -q http://ciscobinary.openh264.org/libopenh264-${OPENH264_VERSION}-linux64.7.so.bz2 \
    && bunzip2 libopenh264-${OPENH264_VERSION}-linux64.7.so.bz2 \
    && mv libopenh264-${OPENH264_VERSION}-linux64.7.so /usr/local/lib/libopenh264.so.7 \
    && ldconfig

# fix libtiff (imei doesn't install the correct version)
RUN apt-get update && apt-get purge -y \
    libtiff-dev libtiff-tools && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp
RUN wget https://download.osgeo.org/libtiff/tiff-4.7.0.tar.gz && \
    tar -xzf tiff-4.7.0.tar.gz && \
    cd tiff-4.7.0 && \
    ./configure && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig
RUN rm -rf /tmp/tiff-4.7.0* 

# run imagemagick easy installer
RUN t=$(mktemp) && \
    wget 'https://dist.1-2.dev/imei.sh' -qO "$t" && \
    bash "$t" && \
    rm "$t"

RUN identify -version

# Dependencies for home-index-thumbnail
RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    ffmpeg \
    webp \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/home_index_thumbnail .

ENTRYPOINT ["python3", "/app/main.py"]
