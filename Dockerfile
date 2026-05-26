FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface
ENV MAX_JOBS=2
ENV NVCC_THREADS=1
ENV WORKER_MODE=single
ENV WORKER_POLL_SECONDS=30
ENV JOB_KEY=jobs/active/current.json

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

RUN cat > /worker.py <<'PY'
import json
import os
import subprocess
import time
from pathlib import Path

import boto3
import numpy as np
from botocore.client import Config
from PIL import Image, ImageDraw


WORKSPACE = Path("/workspace")
SANA_DIR = WORKSPACE / "Sana"
TEST_DIR = WORKSPACE / "job-input"
RESULTS_DIR = WORKSPACE / "results"


def required_env():
    required = [
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET",
        "R2_ENDPOINT",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_json(s3, bucket, key, data):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )


def load_job(s3, bucket, key):
    result = s3.get_object(Bucket=bucket, Key=key)
    raw = result["Body"].read().decode("utf-8")
    return json.loads(raw)


def create_fallback_image(prompt, reference_style, output_path):
    text = (prompt + " " + reference_style).lower()

    width, height = 1280, 704
    img = Image.new("RGB", (width, height), (20, 30, 45))
    draw = ImageDraw.Draw(img)

    if "mountain" in text or "ocean" in text or "sun" in text or "drone" in text:
        # Sky gradient
        for y in range(height):
            r = int(40 + y * 0.08)
            g = int(80 + y * 0.05)
            b = int(125 + y * 0.03)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Sun
        draw.ellipse((930, 80, 1080, 230), fill=(255, 198, 90))

        # Ocean
        draw.rectangle((0, 430, width, height), fill=(25, 95, 135))
        for y in range(455, height, 28):
            draw.line([(0, y), (width, y + 10)], fill=(105, 165, 190), width=2)

        # Mountains
        draw.polygon([(0, 430), (220, 190), (450, 430)], fill=(55, 75, 80))
        draw.polygon([(290, 430), (570, 150), (860, 430)], fill=(45, 68, 78))
        draw.polygon([(720, 430), (980, 210), (1280, 430)], fill=(50, 78, 88))

        # Highlights
        draw.polygon([(220, 190), (300, 430), (450, 430)], fill=(70, 95, 92))
        draw.polygon([(570, 150), (650, 430), (860, 430)], fill=(70, 95, 100))
    else:
        draw.rectangle((0, 450, width, height), fill=(35, 70, 45))
        draw.ellipse((520, 230, 760, 470), fill=(90, 150, 110))

    draw.text((40, 40), "SANA-WM job start image", fill=(255, 255, 255))
    img.save(output_path)


def download_reference_image_if_exists(s3, bucket, job, output_path):
    reference_key = job.get("reference_image_key")
    if not reference_key:
        return False

    print(f"Downloading reference image from R2: {reference_key}", flush=True)
    obj = s3.get_object(Bucket=bucket, Key=reference_key)
    output_path.write_bytes(obj["Body"].read())
    return True


def movement_to_action(job):
    if job.get("action"):
        return job["action"]

    movement = job.get("camera_movement", "slow_forward_drone")

    mapping = {
        "slow_forward_drone": "w-8",
        "gentle_pan_right": "rw-8",
        "gentle_orbit_left": "lw-8",
        "gentle_orbit_right": "rw-8",
        "slow_pull_back": "s-8",
    }

    return mapping.get(movement, "w-8")


def choose_generation_settings(job):
    duration_seconds = int(job.get("duration_seconds", 9))

    if duration_seconds >= 60:
        return {
            "target_seconds": 60,
            "num_frames": int(os.environ.get("FULL_NUM_FRAMES", "321")),
            "step": int(os.environ.get("FULL_STEP", "20")),
            "no_refiner": os.environ.get("USE_REFINER", "false").lower() != "true",
        }

    return {
        "target_seconds": max(1, duration_seconds),
        "num_frames": int(os.environ.get("TINY_NUM_FRAMES", "9")),
        "step": int(os.environ.get("TINY_STEP", "2")),
        "no_refiner": True,
    }


def run_command(command):
    print("Running command:", " ".join(command), flush=True)
    process = subprocess.run(command)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def find_mp4():
    mp4_files = sorted(RESULTS_DIR.rglob("*.mp4"))
    if not mp4_files:
        raise RuntimeError("No MP4 file found in results directory")
    return mp4_files[0]


def get_video_duration_seconds(video_path):
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(completed.stdout.strip())
    except Exception as exc:
        print(f"Could not read video duration, skipping retime: {exc}", flush=True)
        return None


def retime_video_if_needed(video_path, target_seconds):
    if target_seconds < 30:
        return video_path

    duration = get_video_duration_seconds(video_path)
    if not duration:
        return video_path

    print(f"Generated video duration: {duration:.2f}s. Target: {target_seconds}s", flush=True)

    if abs(duration - target_seconds) <= 2:
        print("Video duration is close enough. No retime needed.", flush=True)
        return video_path

    factor = target_seconds / duration
    output_path = video_path.with_name(video_path.stem + "_retimed_60s.mp4")

    print(f"Retiming video with factor {factor:.4f}", flush=True)

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-filter:v",
            f"setpts={factor}*PTS",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )

    return output_path


def process_job():
    required_env()

    bucket = os.environ["R2_BUCKET"]
    job_key = os.environ.get("JOB_KEY", "jobs/active/current.json")

    s3 = s3_client()

    print(f"Reading job from R2: s3://{bucket}/{job_key}", flush=True)
    job = load_job(s3, bucket, job_key)

    job_id = job.get("job_id") or ("sana-wm-job-" + time.strftime("%Y%m%d-%H%M%S"))
    prompt = job.get("prompt") or "A peaceful cinematic nature scene."
    reference_style = job.get("reference_style") or ""

    print(f"Loaded job_id: {job_id}", flush=True)
    print(f"Prompt: {prompt}", flush=True)

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start_image = TEST_DIR / "start.png"
    prompt_file = TEST_DIR / "prompt.txt"
    intrinsics_file = TEST_DIR / "intrinsics.npy"

    used_reference = download_reference_image_if_exists(s3, bucket, job, start_image)

    if not used_reference:
        print("No reference image found in job. Creating fallback start image from prompt/reference style.", flush=True)
        create_fallback_image(prompt, reference_style, start_image)

    prompt_file.write_text(prompt, encoding="utf-8")
    np.save(intrinsics_file, np.array([1000.0, 1000.0, 640.0, 352.0], dtype=np.float32))

    action = movement_to_action(job)
    settings = choose_generation_settings(job)

    print("Generation settings:", settings, flush=True)
    print("Action:", action, flush=True)

    command = [
        "python",
        "inference_video_scripts/inference_sana_wm.py",
        "--image",
        str(start_image),
        "--prompt",
        str(prompt_file),
        "--intrinsics",
        str(intrinsics_file),
        "--action",
        action,
        "--num_frames",
        str(settings["num_frames"]),
        "--step",
        str(settings["step"]),
        "--output_dir",
        str(RESULTS_DIR),
        "--name",
        job_id,
    ]

    if settings["no_refiner"]:
        command.append("--no_refiner")

    run_command(command)

    mp4_file = find_mp4()
    print(f"Generated MP4: {mp4_file}", flush=True)

    final_mp4 = retime_video_if_needed(mp4_file, settings["target_seconds"])
    print(f"Final MP4 for upload: {final_mp4}", flush=True)

    output_prefix = job.get("output_prefix") or f"videos/sana-wm/outputs/{job_id}/"
    if not output_prefix.endswith("/"):
        output_prefix += "/"

    video_key = output_prefix + "video.mp4"
    metadata_key = job.get("metadata_key") or f"learning/job-history/{job_id}.json"
    result_key = f"jobs/completed/{job_id}.json"
    active_result_key = "jobs/active/current-result.json"

    print(f"Uploading video to R2: {video_key}", flush=True)
    s3.upload_file(
        str(final_mp4),
        bucket,
        video_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    temporary_link = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": video_key},
        ExpiresIn=604800,
    )

    completed_job = {
        **job,
        "job_id": job_id,
        "status": "completed",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "SANA-WM",
        "video_key": video_key,
        "metadata_key": metadata_key,
        "result_key": result_key,
        "temporary_video_link": temporary_link,
        "used_reference_image": used_reference,
        "action": action,
        "settings": settings,
    }

    upload_json(s3, bucket, metadata_key, completed_job)
    upload_json(s3, bucket, result_key, completed_job)
    upload_json(s3, bucket, active_result_key, completed_job)

    print("SANA_WM_JOB_SUCCESS", flush=True)
    print("R2_UPLOAD_SUCCESS", flush=True)
    print("R2_VIDEO_KEY=" + video_key, flush=True)
    print("R2_METADATA_KEY=" + metadata_key, flush=True)
    print("R2_RESULT_KEY=" + result_key, flush=True)
    print("R2_TEMP_VIDEO_LINK=" + temporary_link, flush=True)


def main():
    mode = os.environ.get("WORKER_MODE", "single").lower()
    poll_seconds = int(os.environ.get("WORKER_POLL_SECONDS", "30"))

    while True:
        try:
            process_job()
        except Exception as exc:
            print("SANA_WM_JOB_FAILED", flush=True)
            print(str(exc), flush=True)

        if mode != "loop":
            print("Worker finished one attempt. Staying alive so logs remain visible.", flush=True)
            while True:
                time.sleep(3600)

        print(f"Worker loop mode enabled. Sleeping {poll_seconds}s before checking again.", flush=True)
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
PY

RUN cat > /start.sh <<'SH'
#!/bin/bash
echo "Starting reusable SANA-WM R2 job worker..."
echo "This worker reads the job from R2 and generates the requested video."

cd /workspace/Sana

source /opt/conda/etc/profile.d/conda.sh

export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
export HF_HOME=/workspace/.cache/huggingface
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

echo "Accepting Conda terms..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if conda env list | awk "{print \$1}" | grep -qx "sana"; then
  echo "SANA environment already exists. Skipping install."
else
  echo "Installing SANA-WM using official installer..."
  bash ./environment_setup.sh sana
fi

conda activate sana

echo "Installing worker helper packages..."
pip install imageio imageio-ffmpeg boto3 pillow

echo "Checking GPU and imports..."
python - <<PY
import torch
import imageio.v3 as iio
import boto3
from PIL import Image
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Worker imports: OK")
PY

echo "Running worker..."
python /worker.py
SH

RUN chmod +x /start.sh

CMD ["/start.sh"]
