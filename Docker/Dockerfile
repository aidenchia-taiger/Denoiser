FROM ubuntu:18.04

RUN apt-get update \
&& apt-get install --no-install-recommends -y \
# install essentials and build dependencies
    build-essential \
    autotools-dev \
    cmake \
    curl \
    g++ \
    git \
    gfortran-multilib \
    libavcodec-dev \
    libavformat-dev \
    libjpeg-dev \
    libpng-dev \
    liblapacke-dev \
    libswscale-dev \
    libtiff-dev \
    pkg-config \
    libfreetype6-dev \
    libzmq3-dev \
    rsync \
    software-properties-common \
    unzip \
    wget \
    zlib1g-dev \
    libffi-dev \
    # Protobuf
    ca-certificates \
    less \
    procps \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    python3-wheel \
    pkg-config \
# requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
# requirements for TesserOCR
	tesseract-ocr \
	libtesseract-dev \
	libleptonica-dev \
    automake \
    libtool \
    autoconf \
    libpcre3-dev \
    vim \
# Install Python dependencies
&& pip3 install --upgrade pip \
&& pip --no-cache-dir install \
	Pillow==5.1.0 \
	numpy==1.14.2 \
	opencv-python==4.1.0.25 \
	matplotlib \
	scipy \
    scikit-image \
	scikit-learn \
	tesserocr \
	cython \
	IPython[all] \
&& ln -s /usr/bin/pip3 /usr/bin/pip \
&& ln -s /usr/bin/python3 /usr/bin/python \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/* \
&& jupyter-notebook --generate-config \
&& echo "c = get_config()" >> /root/.jupyter/jupyter_notebook_config.py \
&& echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
&& echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py \
&& echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py \
&& echo "alias c='clear'" >> /root/.bashrc

# Allow UI display through host machine
ENV QT_X11_NO_MITSHM 1

# Expose port 8080 for Jupyter notebook
EXPOSE 8080

