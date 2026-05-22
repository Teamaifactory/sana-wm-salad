FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates ffmpeg build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace

RUN git clone https://github.com/NVlabs/Sana.git

RUN printf '%s\n' \
'#!/bin/bash' \
'set -e' \
'echo "Starting SANA-WM setup on SaladCloud..."' \
'cd /workspace/Sana' \
'source /opt/conda/etc/profile.d/conda.sh' \
'if conda env list | awk "{print \$1}" | grep -qx "sana"; then' \
'  echo "SANA environment already exists. Skipping install."' \
'else' \
'  echo "Installing SANA-WM now. This can take a while on first start."' \
'  bash ./environment_setup.sh sana' \
'fi' \
'conda activate sana' \
'echo "SANA-WM container is ready."' \
'python - <<PY' \
'import torch' \
'print("CUDA available:", torch.cuda.is_available())' \
'PY' \
'sleep infinity' \
> /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
