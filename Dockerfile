# Use a base image with Python
FROM ubuntu:22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Install essential tools + libgl1 (needed for opencv/some graphics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bash \
    curl \
    git \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN mkdir -p /miniconda3 && \
    wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh && \
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 && \
    rm /miniconda3/miniconda.sh
ENV PATH=/miniconda3/bin:$PATH

# Accept conda TOS and create environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null; \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null; \
    conda create -n flaxdiff python=3.12 -y

# Activate conda environment and install packages
# Note: This RUN command executes in a single shell layer to keep the conda env active
# Install packages directly into the flaxdiff environment using conda run
RUN conda run --no-capture-output -n flaxdiff pip install --no-cache-dir jax[tpu]==0.5.3 flax[all] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    conda run --no-capture-output -n flaxdiff pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu && \
    conda run --no-capture-output -n flaxdiff pip install --no-cache-dir \
        "diffusers==0.29.2" \
        orbax \
        optax \
        clu \
        grain \
        augmax \
        albumentations \
        datasets \
        "transformers==4.41.2" \
        opencv-python \
        pandas \
        tensorflow-datasets \
        jupyterlab \
        python-dotenv \
        scikit-learn \
        termcolor \
        wrapt \
        wandb \
        gcsfs \
        decord \
        video-reader-rs \
        colorlog \
        importlib_resources \
        flaxdiff && \
    # Clean conda cache afterwards
    conda clean -afy

# Set TOKENIZERS_PARALLELISM env var (will be inherited by processes)
ENV TOKENIZERS_PARALLELISM=false

# Clone FlaxDiff repo for training.py and latest model code (overrides pip flaxdiff)
RUN git clone --depth 1 https://github.com/AshishKumar4/FlaxDiff.git /app/flaxdiff
WORKDIR /app/flaxdiff
RUN conda run --no-capture-output -n flaxdiff pip install --no-cache-dir -e .

# (Optional) Set a default entrypoint - Vertex AI will override this
# ENTRYPOINT ["python", "/app/training.py"]
# For wandb sweeps, the command will be `wandb agent ...`

# Ensure the conda environment is activated for subsequent commands/entrypoint
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flaxdiff"]
# CMD will be provided by Vertex AI job spec (e.g., ["wandb", "agent", "..."])