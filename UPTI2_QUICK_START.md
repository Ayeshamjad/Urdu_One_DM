# UPTI2 Urdu Training - Quick Start Guide

## What Has Been Prepared

I've created a complete training pipeline for your UPTI2 Urdu dataset. Here's what's ready:

### Files Created:
1. **prepare_upti2_dataset.py** - Organizes your UPTI2 dataset into oneDM format
2. **preprocess_upti2_images.py** - Resizes images and creates Laplace edge maps
3. **verify_upti2_setup.py** - Verifies everything is set up correctly
4. **configs/UPTI2_urdu.yml** - Training configuration for UPTI2
5. **run_upti2_training.sh** - Automated script to run everything
6. **UPTI2_TRAINING_GUIDE.md** - Detailed documentation

### Code Changes:
- **data_loader/loader_ara.py** - Updated to support Urdu-specific characters (ٹ ڈ ڑ ں ے پ چ ژ ک گ ی ہ)

## Quick Start (3 Steps)

### Option A: Automated (Recommended)

```bash
# 1. Copy all files to the server
scp -r /home/osama/Desktop/Ayesha/oneDM/Arabic-One-DM/* user@server:/storage/1/saima/oneDM/Arabic-One-DM/

# 2. SSH to server and run
ssh user@server
cd /storage/1/saima/oneDM/Arabic-One-DM
chmod +x run_upti2_training.sh
bash run_upti2_training.sh
```

### Option B: Manual (Step by Step)

```bash
# On the server at /storage/1/saima/oneDM/Arabic-One-DM

# Step 1: Prepare dataset (organize + limit to 20K samples)
python prepare_upti2_dataset.py \
  --image_base /storage/1/saima/images_upti2_2/images \
  --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth \
  --output_base /storage/1/saima/oneDM/upti2_raw \
  --max_train 20000 \
  --max_val 2000

# Step 2: Preprocess images (resize to 64px height + create Laplace edges)
python preprocess_upti2_images.py \
  --data_dir /storage/1/saima/oneDM/upti2_raw \
  --output_dir /storage/1/saima/oneDM/upti2_processed \
  --threads 16

# Step 3: Copy annotation files
mkdir -p data
cp /storage/1/saima/oneDM/upti2_raw/train.txt data/
cp /storage/1/saima/oneDM/upti2_raw/val.txt data/

# Step 4: Verify setup
python verify_upti2_setup.py

# Step 5: Start training
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_training
```

## What Each Step Does

### Step 1: prepare_upti2_dataset.py
- Reads your UPTI2 images from nested structure (train/1/Font/degradation/1.png)
- Matches with ground truth files (groundtruth/train/1.txt)
- Creates unique writer IDs: `FontName_degradation_imagenum`
- Randomly samples 20,000 training images
- Copies images to flat structure: `upti2_raw/train/FontName_degradation_1.png`
- Creates `train.txt`: `writer_id,image_name.png transcription`

**Expected output:**
```
/storage/1/saima/oneDM/upti2_raw/
├── train/
│   ├── AlviNastaleeq_nodegradation_1.png
│   ├── AlviNastaleeq_nodegradation_2.png
│   └── ... (20,000 images)
├── val/
│   └── ... (2,000 images)
├── train.txt (20,000 lines)
└── val.txt (2,000 lines)
```

### Step 2: preprocess_upti2_images.py
- Resizes all images to height=64 pixels (width scaled proportionally)
- Ensures width is multiple of 16 (required by model)
- Creates Laplace edge maps for style conditioning
- Saves to `upti2_processed/` and `upti2_processed_laplace/`

### Step 3: Copy annotations
- Moves train.txt and val.txt to `data/` folder where model expects them

### Step 4: verify_upti2_setup.py
- Checks all files are in place
- Verifies image dimensions
- Confirms annotation format is correct
- Ensures images referenced in train.txt actually exist

### Step 5: Start training
- Loads config from `configs/UPTI2_urdu.yml`
- Downloads Stable Diffusion VAE (automatic)
- Starts distributed training (even on single GPU)
- Saves checkpoints to `/storage/1/saima/oneDM/output_upti2/`

## Monitoring Training

```bash
# Watch training progress
tail -f /storage/1/saima/oneDM/output_upti2/logs/upti2_training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check saved checkpoints
ls -lh /storage/1/saima/oneDM/output_upti2/checkpoints/
```

## Configuration Details

File: `configs/UPTI2_urdu.yml`

Key settings:
- **Batch size**: 32 (reduce to 16 or 8 if out of memory)
- **Epochs**: 315
- **Learning rate**: 0.0001
- **Image height**: 64 pixels
- **Data paths**: Point to `/storage/1/saima/oneDM/upti2_processed`

## Dataset Statistics

After running Step 1, you'll have:
- **Training samples**: 20,000 Urdu handwriting images
- **Validation samples**: 2,000 images
- **Unique fonts**: 4 (Alvi, Jameel Noori, Nafees, Pak Nastaleeq)
- **Degradation levels**: 4 per font (high, low, medium, nodegradation)
- **Total unique writer IDs**: 4 fonts × 4 degradations = 16 "writers"

## Urdu Character Support

The model now supports these Urdu-specific characters:
- **ٹ** (Ṭe) - contextual forms
- **ڈ** (Ḍal) - non-joining
- **ڑ** (Ṛe) - non-joining
- **ں** (Noon Ghunna) - non-joining
- **ے** (Ye) - non-joining
- **پ** (Pe) - contextual forms
- **چ** (Che) - contextual forms
- **ژ** (Zhe) - non-joining
- **ک** (Kaf) - contextual forms
- **گ** (Gaf) - contextual forms
- **ی** (Ye) - contextual forms
- **ہ** (Gol He) - contextual forms

Plus all standard Arabic characters.

## Troubleshooting

### "Out of memory" error
```bash
# Edit configs/UPTI2_urdu.yml
# Change: IMS_PER_BATCH: 16  (or 8)
```

### "No such file or directory"
```bash
# Run verification script
python verify_upti2_setup.py
# It will tell you exactly what's missing
```

### "Cannot find unifont.pickle"
```bash
# The existing unifont.pickle should work
# Make sure it exists in data/unifont.pickle
ls -lh data/unifont.pickle
```

### Training is very slow
- Check GPU usage: `nvidia-smi`
- Reduce NUM_THREADS in config if CPU is bottleneck
- Use multiple GPUs if available

### Validation loss not decreasing
- Make sure you have diverse samples in training set
- Check if images are properly preprocessed (height=64)
- Verify ground truth text matches images

## Expected Training Time

With UPTI2 (20,000 samples):
- **1 GPU (RTX 3090)**: ~3-5 days for 315 epochs
- **2 GPUs**: ~2-3 days
- **4 GPUs**: ~1-2 days

First epoch will be slow (model initialization), subsequent epochs will be faster.

## What Happens During Training

1. **Epoch 0-20**: Model learns basic Urdu character shapes
2. **Epoch 20-100**: Model learns different writing styles (fonts)
3. **Epoch 100-200**: Model learns to copy style from reference images
4. **Epoch 200-315**: Fine-tuning and quality improvement

Checkpoints are saved every 5 epochs in:
```
/storage/1/saima/oneDM/output_upti2/checkpoints/
```

## After Training

To test your trained model:

```bash
python test.py \
  --cfg configs/UPTI2_urdu.yml \
  --model_path /storage/1/saima/oneDM/output_upti2/checkpoints/epoch_315.pth \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --output_dir /storage/1/saima/oneDM/results_upti2
```

## Files to Transfer to Server

You need to copy these files from your local machine to the server:

```
Arabic-One-DM/
├── prepare_upti2_dataset.py          (NEW)
├── preprocess_upti2_images.py        (NEW)
├── verify_upti2_setup.py             (NEW)
├── run_upti2_training.sh             (NEW)
├── check_urdu_coverage.py            (NEW)
├── configs/UPTI2_urdu.yml            (NEW)
├── data_loader/loader_ara.py         (MODIFIED for Urdu)
└── [all other existing files]
```

## Summary Commands

```bash
# Complete pipeline in one go:
cd /storage/1/saima/oneDM/Arabic-One-DM
bash run_upti2_training.sh

# Or run each step manually as shown in "Option B" above
```

## Need Help?

Check these first:
1. Run `python verify_upti2_setup.py` to diagnose issues
2. Check log file: `tail -f /storage/1/saima/oneDM/output_upti2/logs/upti2_training.log`
3. Verify GPU: `nvidia-smi`
4. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## Important Notes

1. **Dataset size**: Script will automatically limit to 20,000 training samples
2. **Urdu support**: Code has been updated to handle Urdu-specific characters
3. **GPU required**: Training requires CUDA-capable GPU (8GB+ VRAM recommended)
4. **Disk space**: Processed dataset will need ~5-10GB depending on image sizes
5. **Internet required**: First run downloads Stable Diffusion VAE (~4GB)

Good luck with your training! 🚀
