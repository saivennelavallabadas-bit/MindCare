
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    wget \
    unzip \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libboost-all-dev \
    libtbb-dev \
    libopencv-dev \
    libdlib-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/TadasBaltrusaitis/OpenFace.git /opt/OpenFace \
    && cd /opt/OpenFace \
    && bash download_models.sh \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE .. \
    && cmake --build . --parallel 4

ENV OPENFACE_PATH=/opt/OpenFace/build/bin/FeatureExtraction

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
