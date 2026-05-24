FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

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
'echo "Starting SANA-WM demo smoke test..."' \
'cd /workspace/Sana' \
'source /opt/conda/etc/profile.d/conda.sh' \
'export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes' \
'export HF_HOME=/workspace/.cache/huggingface' \
'echo "Accepting Conda terms..."' \
'conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true' \
'conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true' \
'if conda env list | awk "{print \$1}" | grep -qx "sana"; then' \
'  echo "SANA environment already exists. Skipping install."' \
'else' \
'  echo "Installing SANA-WM using official installer..."' \
'  bash ./environment_setup.sh sana' \
'fi' \
'conda activate sana' \
'echo "Checking GPU..."' \
'python - <<PY' \
'import torch' \
'print("CUDA available:", torch.cuda.is_available())' \
'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' \
'PY' \
'echo "Creating test image, prompt, and camera intrinsics..."' \
'mkdir -p /workspace/test /workspace/results' \
'python - <<PY' \
'from PIL import Image, ImageDraw' \
'import numpy as np' \
'img = Image.new("RGB", (1280, 704), (30, 45, 65))' \
'd = ImageDraw.Draw(img)' \
'd.ellipse((520, 230, 760, 470), fill=(90, 150, 110))' \
'd.rectangle((0, 500, 1280, 704), fill=(35, 70, 45))' \
'd.text((40, 40), "SANA-WM smoke test", fill=(255, 255, 255))' \
'img.save("/workspace/test/start.png")' \
'open("/workspace/test/prompt.txt", "w").write("A calm cinematic scene of a small green turtle standing in a forest clearing, soft morning light, realistic, peaceful atmosphere.")' \
'np.save("/workspace/test/intrinsics.npy", np.array([1000.0, 1000.0, 640.0, 352.0], dtype=np.float32))' \
'PY' \
'echo "Running tiny SANA-WM generation test..."' \
'python inference_video_scripts/inference_sana_wm.py \
  --image /workspace/test/start.png \
  --prompt /workspace/test/prompt.txt \
  --intrinsics /workspace/test/intrinsics.npy \
  --action "w-8" \
  --num_frames 9 \
  --step 2 \
  --no_refiner \
  --output_dir /workspace/results \
  --name smoke' \
'RESULT=$?' \
'if [ "$RESULT" -ne 0 ]; then' \
'  echo "SANA-WM DEMO TEST FAILED."' \
'  echo "Read the logs above for the exact error."' \
'  sleep infinity' \
'fi' \
'echo "SANA-WM DEMO TEST PASSED."' \
'echo "Generated files:"' \
'ls -lah /workspace/results' \
'echo "Expected output: /workspace/results/smoke_generated.mp4"' \
'echo "Container will stay alive so logs remain visible."' \
'sleep infinity' \
> /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
