############################
# base: solver + python 3.11
############################
FROM ubuntu:20.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

# System deps + deadsnakes for Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget unzip curl git \
      software-properties-common \
      libstdc++6 libgomp1 libgcc-s1 libgfortran5 \
      build-essential python3-dev \
      libffi-dev libssl-dev pkg-config cmake \
      rustc cargo \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
 && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python3.11 /tmp/get-pip.py \
 && rm -rf /var/lib/apt/lists/* /tmp/get-pip.py

# Make python/pip point to 3.11 explicitly
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 \
 && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# --- (optional) Install prebuilt TexasSolver so it's available in this image ---
WORKDIR /opt/texas-solver
ARG TEXASSOLVER_VERSION=v0.2.0
RUN wget -q https://github.com/bupticybee/TexasSolver/releases/download/${TEXASSOLVER_VERSION}/TexasSolver-${TEXASSOLVER_VERSION}-Linux.zip -O /tmp/solver.zip \
 && unzip -q /tmp/solver.zip \
 && mv TexasSolver-*-Linux/* . \
 && rm -rf /tmp/solver.zip __MACOSX
ENV PATH="/opt/texas-solver:${PATH}"

# Global thread sanity
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Python deps
WORKDIR /app
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir \
      "Cython>=0.29.36" \
      "numpy==1.26.4"
COPY requirements.txt /app/requirements.txt

# If you use torch and want CPU wheels, consider pinning or pre-installing like:
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -v -r /app/requirements.txt

# Copy the project
COPY . /app

############################
# worker target (default “do nothing”; cloud-init will run the script)
############################
FROM base AS worker
WORKDIR /app
CMD ["sleep", "infinity"]

############################
# api target (if/when you want the API image)
############################
FROM base AS api
WORKDIR /app
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]