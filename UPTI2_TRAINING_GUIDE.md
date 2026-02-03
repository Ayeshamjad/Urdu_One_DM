# Complete Guide: Training oneDM on UPTI2 Urdu Dataset

This guide provides step-by-step instructions for training the oneDM model on the UPTI2 Urdu handwriting dataset.

## Dataset Structure

Your UPTI2 dataset has:
- **Images**: `/storage/1/saima/images_upti2_2/images/train`
  - Structure: `train/[number]/[Font Name]/[degradation]/[number].png`
  - Fonts: Alvi Nastaleeq, Jameel Noori Nastaleeq, Nafees Nastaleeq, Pak Nastaleeq
  - Degradation levels: high, low, medium, nodegradation

- **Ground Truth**: `/home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth/train`
  - Format: `[number].txt` contains the Urdu text for corresponding image

## Prerequisites

Make sure you have the required packages installed:

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM
pip install -r requirements.txt
```

## Step-by-Step Training Process

### Step 1: Prepare Dataset (Create train.txt and organize images)

This step organizes your UPTI2 dataset into the format expected by oneDM and limits it to 20,000 training samples.

**SINGLE LINE COMMAND (copy this entire line):**
```bash
python prepare_upti2_dataset.py --image_base /storage/1/saima/images_upti2_2/images --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth --output_base /storage/1/saima/oneDM/upti2_raw --max_train 20000 --max_val 2000 --seed 42
```

**Or use the shell script:**
```bash
./run_prepare.sh
```

**What this does:**
- Reads all images and ground truth files
- Creates unique writer IDs based on font and degradation level
- Randomly selects 20,000 samples for training
- Creates 2,000 samples for validation
- Copies images to `/storage/1/saima/oneDM/upti2_raw/train/` and `val/`
- Creates `train.txt` and `val.txt` in the format: `writer_id,image_name transcription`

**Expected output:**
```
/storage/1/saima/oneDM/upti2_raw/
├── train/
│   ├── AlviNastaleeq_nodeg_1.png
│   ├── AlviNastaleeq_nodeg_2.png
│   └── ...
├── val/
│   └── ...
├── train.txt
└── val.txt
```

### Step 2: Preprocess Images (Resize and Create Laplace Edges)

This step resizes all images to height=64 pixels and creates Laplace edge maps for style conditioning.

```bash
python preprocess_upti2_images.py \
  --data_dir /storage/1/saima/oneDM/upti2_raw \
  --output_dir /storage/1/saima/oneDM/upti2_processed \
  --threads 16
```

**What this does:**
- Resizes all images to height=64 pixels (width scaled proportionally)
- Creates Laplace edge detection images for style conditioning
- Saves to `/storage/1/saima/oneDM/upti2_processed/` and `upti2_processed_laplace/`

**Expected output:**
```
/storage/1/saima/oneDM/upti2_processed/
├── train/ (resized images)
└── val/

/storage/1/saima/oneDM/upti2_processed_laplace/
├── train/ (edge maps)
└── val/
```

### Step 3: Copy Annotation Files

Copy the train.txt and val.txt to the data directory where the model expects them:

```bash
mkdir -p /storage/1/saima/oneDM/Arabic-One-DM/data
cp /storage/1/saima/oneDM/upti2_raw/train.txt /storage/1/saima/oneDM/Arabic-One-DM/data/
cp /storage/1/saima/oneDM/upti2_raw/val.txt /storage/1/saima/oneDM/Arabic-One-DM/data/
```

### Step 4: Download Required Pretrained Models

You need to download pretrained models before training:

**a) Stable Diffusion VAE:**
```bash
# This will be downloaded automatically from Hugging Face when you run training
# Make sure you have internet access or download manually
```

**b) (Optional) Pretrained ResNet-18 for Arabic:**
If you have a pretrained ResNet-18 model from Khat² dataset training, place it at:
```
/storage/1/saima/oneDM/models/resnet18_khat_pretrained.pth
```

If you don't have it, the model will train from scratch (takes longer but still works).

### Step 5: Configure Training

The configuration file has been created at `configs/UPTI2_urdu.yml` with these settings:

- **Batch size**: 32 (adjust based on your GPU memory)
- **Learning rate**: 0.0001
- **Epochs**: 315
- **Image paths**: Point to your processed UPTI2 data
- **Output**: `/storage/1/saima/oneDM/output_upti2`

**To adjust batch size** (if you get out-of-memory errors):
```bash
nano configs/UPTI2_urdu.yml
# Change IMS_PER_BATCH to 16 or 8
```

### Step 6: Start Training

**For multi-GPU training** (recommended if you have multiple GPUs):

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

# Set number of GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3  # adjust based on available GPUs

# Start distributed training
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --feat_model /storage/1/saima/oneDM/models/resnet18_khat_pretrained.pth \
  --log upti2_training
```

**For single GPU training:**

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_training
```

**If you DON'T have pretrained ResNet-18**, remove the `--feat_model` argument:
```bash
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_training
```

### Step 7: Monitor Training

Training logs and checkpoints will be saved to:
```
/storage/1/saima/oneDM/output_upti2/
├── logs/
│   └── upti2_training.log
└── checkpoints/
    ├── epoch_005.pth
    ├── epoch_010.pth
    └── ...
```

Monitor training progress:
```bash
tail -f /storage/1/saima/oneDM/output_upti2/logs/upti2_training.log
```

## Important Parameters to Know

### In `configs/UPTI2_urdu.yml`:

- **IMS_PER_BATCH**: Batch size = 32 (reduce to 16 or 8 if out of memory)
- **EPOCHS**: 200 (reduced from 315 for 20K subset)
- **SNAPSHOT_ITERS**: 25 (saves model every 25 epochs)
- **VALIDATE_ITERS**: 625 (~1 epoch, generates sample images)
- **BASE_LR**: Learning rate (0.0001 is good default)
- **NUM_THREADS**: 16 (data loading threads)

### Space-Saving Features ✅

- **Auto-cleanup**: Only keeps last 3 checkpoints (saves ~74GB of space!)
- **Smart saving**: Checkpoints every 25 epochs instead of 5
- **Total storage needed**: ~6GB instead of ~80GB

### In training command:

- **--stable_dif_path**: Path to Stable Diffusion pretrained VAE
- **--feat_model**: Path to pretrained ResNet-18 (optional)
- **--one_dm**: Path to pretrained oneDM model (for fine-tuning)
- **--noise_offset**: Noise strength (default 0, can increase for more variation)

## Troubleshooting

### Out of Memory Error:
```bash
# Reduce batch size in configs/UPTI2_urdu.yml
IMS_PER_BATCH: 16  # or 8
```

### "No module named 'diffusers'":
```bash
pip install diffusers transformers accelerate
```

### CUDA/cuDNN errors:
```bash
# Make sure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### Dataset not found:
- Check all paths in `configs/UPTI2_urdu.yml`
- Verify `data/train.txt` and `data/val.txt` exist
- Verify preprocessed images exist in specified directories

## Testing the Trained Model

After training, test your model:

```bash
python test.py \
  --cfg configs/UPTI2_urdu.yml \
  --model_path /storage/1/saima/oneDM/output_upti2/checkpoints/epoch_315.pth \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --output_dir /storage/1/saima/oneDM/results_upti2
```

## Summary of All Commands

```bash
# Step 1: Prepare dataset
cd /storage/1/saima/oneDM/Arabic-One-DM
python prepare_upti2_dataset.py \
  --image_base /storage/1/saima/images_upti2_2/images \
  --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth \
  --output_base /storage/1/saima/oneDM/upti2_raw \
  --max_train 20000 --max_val 2000

# Step 2: Preprocess images
python preprocess_upti2_images.py \
  --data_dir /storage/1/saima/oneDM/upti2_raw \
  --output_dir /storage/1/saima/oneDM/upti2_processed \
  --threads 16

# Step 3: Copy annotations
mkdir -p data
cp /storage/1/saima/oneDM/upti2_raw/train.txt data/
cp /storage/1/saima/oneDM/upti2_raw/val.txt data/

# Step 4: Start training (single GPU)
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI2_urdu.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_training
```

## Expected Training Time

With 200 epochs (updated from 315):
- **With 1 GPU (e.g., RTX 3090)**: ~2-3 days for 200 epochs
- **With 4 GPUs**: ~12-18 hours for 200 epochs
- **Dataset size**: 20,000 samples

## Storage Requirements

- **Old config (315 epochs, save every 5)**: ~80GB (40 checkpoints)
- **New config (200 epochs, auto-cleanup)**: ~6GB (only last 3 checkpoints kept) ✅
- **Sample images**: ~100MB
- **Logs**: ~50MB
- **Total needed**: ~6.2GB

## Questions?

If you encounter any issues, check:
1. GPU memory (`nvidia-smi`)
2. CUDA availability (`python -c "import torch; print(torch.cuda.is_available())"`)
3. Data paths in config file
4. Log files for specific error messages
