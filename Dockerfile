############################
# base: Ubuntu + Python 3.11 + solver (no ML)
############################
FROM ubuntu:20.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget unzip curl \
      software-properties-common \
      libstdc++6 libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm -rf /var/lib/apt/lists/* /tmp/get-pip.py

# Make python/pip default to 3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip

# Install prebuilt TexasSolver
WORKDIR /opt/texas-solver
ARG TEXASSOLVER_VERSION=v0.2.0
RUN wget -q https://github.com/bupticybee/TexasSolver/releases/download/${TEXASSOLVER_VERSION}/TexasSolver-${TEXASSOLVER_VERSION}-Linux.zip -O /tmp/solver.zip \
 && unzip -q /tmp/solver.zip -d /opt/texas-solver/ \
 && rm -rf /tmp/solver.zip __MACOSX \
 && chmod -R 755 /opt/texas-solver \
 && ln -sf /opt/texas-solver/console_solver /usr/local/bin/texas-solver

ENV PATH="/opt/texas-solver:${PATH}" \
    SOLVER_BIN="/usr/local/bin/texas-solver" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

############################
# worker image
############################
FROM base AS worker
WORKDIR /app

# Copy full project
COPY . /app

# Install runtime deps (use minimal worker requirements)
COPY requirements-worker.txt /app/requirements-worker.txt
RUN pip install --no-cache-dir -r /app/requirements-worker.txt \
 && rm -rf /root/.cache/pip

# stay idle; cloud-init will call python tools/rangenet/worker_flop.py
CMD ["sleep","infinity"]

############################
# api image
############################
FROM base AS api
WORKDIR /app

# Copy full project
COPY . /app

# Install runtime deps (use minimal API requirements)
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt \
 && rm -rf /root/.cache/pip

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]