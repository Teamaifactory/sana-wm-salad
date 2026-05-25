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
'echo "Starting SANA-WM full loop test: generate video + upload to R2..."' \
'cd /workspace/Sana' \
'source /opt/conda/etc/profile.d/conda.sh' \
'export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes' \
'export HF_HOME=/workspace/.cache/huggingface' \
'echo "Checking required R2 environment variables..."' \
'for VAR in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ACCOUNT_ID R2_BUCKET R2_ENDPOINT; do' \
'  if [ -z "${!VAR}" ]; then' \
'    echo "Missing required env var: $VAR"' \
'    echo "Full loop test cannot continue without R2 credentials."' \
'    sleep infinity' \
'  fi' \
'done' \
'echo "R2 bucket: $R2_BUCKET"' \
'echo "R2 endpoint: $R2_ENDPOINT"' \
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
'echo "Installing upload/video helper packages..."' \
'pip install imageio imageio-ffmpeg boto3' \
'python - <<PY' \
'import torch' \
'import imageio.v3 as iio' \
'import boto3' \
'print("CUDA available:", torch.cuda.is_available())' \
'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' \
'print("imageio import: OK")' \
'print("boto3 import: OK")' \
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
'd.text((40, 40), "SANA-WM R2 full loop test", fill=(255, 255, 255))' \
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
'  echo "SANA-WM VIDEO GENERATION FAILED."' \
'  echo "Read the logs above for the exact error."' \
'  sleep infinity' \
'fi' \
'echo "SANA-WM VIDEO GENERATION PASSED."' \
'echo "Generated local files:"' \
'ls -lah /workspace/results' \
'echo "Finding generated MP4..."' \
'MP4_FILE=$(find /workspace/results -name "*.mp4" | head -n 1)' \
'if [ -z "$MP4_FILE" ]; then' \
'  echo "No MP4 file found. Cannot upload to R2."' \
'  sleep infinity' \
'fi' \
'echo "MP4 file found: $MP4_FILE"' \
'echo "Uploading video and metadata to Cloudflare R2..."' \
'python - <<PY' \
'import os, json, time, boto3' \
'from botocore.client import Config' \
'mp4_file = os.environ.get("MP4_FILE")' \
'if not mp4_file or not os.path.exists(mp4_file):' \
'    raise SystemExit("MP4_FILE missing or does not exist")' \
'bucket = os.environ["R2_BUCKET"]' \
'endpoint = os.environ["R2_ENDPOINT"]' \
'access_key = os.environ["R2_ACCESS_KEY_ID"]' \
'secret_key = os.environ["R2_SECRET_ACCESS_KEY"]' \
'job_id = "sana-wm-smoke-r2-" + time.strftime("%Y%m%d-%H%M%S")' \
'video_key = f"videos/sana-wm/outputs/{job_id}/smoke.mp4"' \
'meta_key = f"learning/job-history/{job_id}.json"' \
's3 = boto3.client(' \
'    "s3",' \
'    endpoint_url=endpoint,' \
'    aws_access_key_id=access_key,' \
'    aws_secret_access_key=secret_key,' \
'    config=Config(signature_version="s3v4"),' \
'    region_name="auto"' \
')' \
's3.upload_file(mp4_file, bucket, video_key, ExtraArgs={"ContentType": "video/mp4"})' \
'metadata = {' \
'    "job_id": job_id,' \
'    "model": "SANA-WM",' \
'    "test_type": "tiny smoke test",' \
'    "video_key": video_key,' \
'    "prompt": "A calm cinematic scene of a small green turtle standing in a forest clearing, soft morning light, realistic, peaceful atmosphere.",' \
'    "action": "w-8",' \
'    "num_frames": 9,' \
'    "status": "uploaded_to_r2"' \
'}' \
's3.put_object(Bucket=bucket, Key=meta_key, Body=json.dumps(metadata, indent=2), ContentType="application/json")' \
'url = s3.generate_presigned_url(' \
'    "get_object",' \
'    Params={"Bucket": bucket, "Key": video_key},' \
'    ExpiresIn=604800' \
')' \
'print("R2_UPLOAD_SUCCESS")' \
'print("R2_VIDEO_KEY=" + video_key)' \
'print("R2_METADATA_KEY=" + meta_key)' \
'print("R2_TEMP_VIDEO_LINK=" + url)' \
'PY' \
'UPLOAD_RESULT=$?' \
'if [ "$UPLOAD_RESULT" -ne 0 ]; then' \
'  echo "R2 UPLOAD FAILED."' \
'  echo "Read the logs above for the exact error."' \
'  sleep infinity' \
'fi' \
'echo "FULL LOOP TEST PASSED: SANA-WM generated video and uploaded it to R2."' \
'echo "Copy the R2_TEMP_VIDEO_LINK above and open it in your browser."' \
'echo "Container will stay alive so logs remain visible."' \
'sleep infinity' \
> /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
