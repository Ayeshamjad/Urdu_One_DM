#!/bin/bash
# Script to re-prepare dataset with ONLY nodegradation images

echo "=========================================="
echo "Re-preparing UPTI2 Dataset (nodeg only)"
echo "=========================================="

cd /storage/1/saima/oneDM/Arabic-One-DM

# Pull latest code (has nodegradation filter)
echo "[1/4] Pulling latest code..."
git pull

# Delete old mixed dataset
echo "[2/4] Removing old mixed dataset..."
rm -rf /storage/1/saima/oneDM/upti2_raw_nodeg
rm -rf /storage/1/saima/oneDM/upti2_processed_nodeg
rm -rf /storage/1/saima/oneDM/upti2_processed_nodeg_laplace

# Re-run dataset preparation (will only use nodegradation)
echo "[3/4] Preparing dataset (nodegradation ONLY)..."
python prepare_upti2_dataset.py \
  --image_base /storage/1/saima/images_upti2_2/images \
  --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth \
  --output_base /storage/1/saima/oneDM/upti2_raw_nodeg \
  --max_train 20000 \
  --max_val 2000 \
  --seed 42

# Verify only nodegradation images
echo "[4/4] Verifying dataset..."
echo ""
echo "Checking first 20 lines of train.txt:"
head -20 /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt

echo ""
echo "Counting degradation types:"
echo "nodegradation: $(grep -c 'nodegradation' /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt)"
echo "low: $(grep -c '_low_' /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt)"
echo "medium: $(grep -c '_medium_' /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt)"
echo "high: $(grep -c '_high_' /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt)"

echo ""
echo "✅ If only nodegradation count is non-zero, dataset is correct!"
echo ""
echo "Next steps:"
echo "1. Run: python preprocess_upti2_images.py (to resize images)"
echo "2. Update config paths to use upti2_processed_nodeg"
echo "3. Start training!"
