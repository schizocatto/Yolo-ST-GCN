#!/bin/bash
# setup.sh
# Bootstrap a blank CUDA 12.4 Linux cloud container.
# Run once: bash setup.sh
#
# Assumptions:
#   - sudo apt available
#   - pip available
#   - CUDA 12.4 already installed by the host image

set -e

echo "=== [1/4] System packages ==="
sudo apt-get update -q
sudo apt-get install -y -q git

echo "=== [2/4] Python dependencies ==="
pip install --upgrade pip -q

# PyTorch built against CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# Project deps (mirrors requirements.txt minus ultralytics/opencv which are optional for training)
pip install numpy scipy pandas scikit-learn matplotlib seaborn tqdm python-dotenv huggingface_hub -q

echo "=== [3/4] Clone repo ==="
REPO_URL="https://github.com/schizocatto/Yolo-ST-GCN.git"
BRANCH="${BRANCH:-refactor-1}"
REPO_DIR="${REPO_DIR:-/workspace/Yolo-ST-GCN}"

if [ ! -d "$REPO_DIR" ]; then
    git clone -b "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR — pulling latest."
    git -C "$REPO_DIR" fetch origin "$BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull origin "$BRANCH"
fi

echo "=== [4/4] Done ==="
echo "Repo : $REPO_DIR"
echo "Run  : cd $REPO_DIR && cp .env.example .env  # then edit .env with your paths"
