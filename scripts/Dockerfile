# -----------------------------
# ✅ Base image with CUDA 12.1 and PyTorch
# -----------------------------
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# -----------------------------
# ✅ Install system packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# ✅ Create working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# ✅ Copy source code
# -----------------------------
COPY . /app

# -----------------------------
# ✅ Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip && \
    pip install \
    pandas \
    tabulate \
    matplotlib \
    tqdm \
    wandb \
    ipython \
    jupyterlab

# Install JRL (from GitHub, pinned to working commit)
RUN pip install git+https://github.com/jstmn/jrl.git@e08a900c4ef77d75a54446c193b4a49acfd3cbaf

# -----------------------------
# ✅ Set environment variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# ✅ Optional (if using wandb)
# -----------------------------
ENV WANDB_MODE=offline

# -----------------------------
# ✅ Run your training script
# -----------------------------
CMD ["python", "train.py"]
