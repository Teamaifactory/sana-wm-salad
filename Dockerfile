FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y \
    git wget curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace

RUN git clone https://github.com/NVlabs/Sana.git

WORKDIR /workspace/Sana

RUN bash ./environment_setup.sh sana

RUN printf '%s\n' \
'#!/bin/bash' \
'source /opt/conda/etc/profile.d/conda.sh' \
'conda activate sana' \
'echo "SANA-WM container started"' \
'python - <<PY' \
'import torch' \
'print("CUDA available:", torch.cuda.is_available())' \
'PY' \
'sleep infinity' \
> /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
