#!/bin/bash
# Complete automated script for UPTI2 training on oneDM
# Run this script from /storage/1/saima/oneDM/Arabic-One-DM

set -e  # Exit on error

echo "=========================================="
echo "UPTI2 Urdu Handwriting Training Pipeline"
echo "=========================================="

# Configuration
IMAGE_BASE="/storage/1/saima/images_upti2_2/images"
GT_BASE="/home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth"
RAW_OUTPUT="/storage/1/saima/oneDM/upti2_raw"
PROCESSED_OUTPUT="/storage/1/saima/oneDM/upti2_processed"
MAX_TRAIN=20000
MAX_VAL=2000
NUM_THREADS=16
NUM_GPUS=1  # Change this based on your available GPUs

# Step 1: Prepare dataset
echo ""
echo "[Step 1/5] Preparing UPTI2 dataset..."
python prepare_upti2_dataset.py \
  --image_base "$IMAGE_BASE" \
  --gt_base "$GT_BASE" \
  --output_base "$RAW_OUTPUT" \
  --max_train $MAX_TRAIN \
  --max_val $MAX_VAL \
  --seed 42

if [ $? -ne 0 ]; then
    echo "Error in dataset preparation!"
    exit 1
fi

# Step 2: Preprocess images
echo ""
echo "[Step 2/5] Preprocessing images (resize + Laplace)..."
python preprocess_upti2_images.py \
  --data_dir "$RAW_OUTPUT" \
  --output_dir "$PROCESSED_OUTPUT" \
  --threads $NUM_THREADS

if [ $? -ne 0 ]; then
    echo "Error in image preprocessing!"
    exit 1
fi

# Step 3: Copy annotations
echo ""
echo "[Step 3/5] Copying annotation files..."
mkdir -p data
cp "$RAW_OUTPUT/train.txt" data/
cp "$RAW_OUTPUT/val.txt" data/

# Verify files exist
if [ ! -f "data/train.txt" ] || [ ! -f "data/val.txt" ]; then
    echo "Error: Annotation files not found!"
    exit 1
fi

echo "Annotation files copied successfully."
echo "Train samples: $(wc -l < data/train.txt)"
echo "Val samples: $(wc -l < data/val.txt)"

# Step 4: Check for pretrained models
echo ""
echo "[Step 4/5] Checking for pretrained models..."

FEAT_MODEL=""
if [ -f "/storage/1/saima/oneDM/models/resnet18_khat_pretrained.pth" ]; then
    echo "Found pretrained ResNet-18 model."
    FEAT_MODEL="--feat_model /storage/1/saima/oneDM/models/resnet18_khat_pretrained.pth"
else
    echo "No pretrained ResNet-18 found. Training from scratch."
fi

# Step 5: Start training
echo ""
echo "[Step 5/5] Starting training..."
echo "Using $NUM_GPUS GPU(s)"

# Set visible GPUs (adjust based on your setup)
export CUDA_VISIBLE_DEVICES=0  # Change to 0,1,2,3 for multiple GPUs

# Launch training
python -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPUS \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  $FEAT_MODEL \
  --log upti2_training

echo ""
echo "=========================================="
echo "Training Started!"
echo "Monitor progress:"
echo "  tail -f /storage/1/saima/oneDM/output_upti2/logs/upti2_training.log"
echo "=========================================="
