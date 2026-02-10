# Complete Guide: Training oneDM on UPTI2 Urdu Dataset (UPDATED)

This guide provides step-by-step instructions for training the oneDM model on the UPTI2 Urdu handwriting dataset.

## 🔧 Recent Bug Fixes (Feb 2026)

**CRITICAL FIXES APPLIED:**
1. ✅ **Dataset filter**: Now uses ONLY `nodegradation` images (high-quality clean text)
2. ✅ **ContentData fix**: Fixed letter2index mapping for proper text conditioning
3. ✅ **Validation fix**: Validation now correctly encodes Urdu text

**If you trained before these fixes**, you need to:
- Re-prepare dataset with nodegradation-only filter
- Restart training from scratch (previous checkpoints won't work correctly)

---

## Dataset Structure

Your UPTI2 dataset has:
- **Images**: `/storage/1/saima/images_upti2_2/images/train`
  - Structure: `train/[number]/[Font Name]/[degradation]/[number].png`
  - Fonts: Alvi Nastaleeq, Jameel Noori Nastaleeq, Nafees Nastaleeq, Pak Nastaleeq
  - Degradation levels: high, low, medium, **nodegradation** ← WE USE ONLY THIS

- **Ground Truth**: `/home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth/train`
  - Format: `[number].txt` contains the Urdu text for corresponding image

---

## Prerequisites

Make sure you have the required packages installed:

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM
pip install -r requirements.txt
```

---

## Step-by-Step Training Process

### Step 1: Prepare Dataset (NODEGRADATION ONLY) ✨

⚠️ **UPDATED**: Now uses ONLY high-quality nodegradation images

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

python prepare_upti2_dataset.py \
  --image_base /storage/1/saima/images_upti2_2/images \
  --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth \
  --output_base /storage/1/saima/oneDM/upti2_raw_nodeg \
  --max_train 20000 \
  --max_val 2000 \
  --seed 42
```

**What this does:**
- ✅ Filters for ONLY `nodegradation` images (clean, high-quality)
- Creates unique writer IDs based on font
- Randomly selects 20,000 samples for training
- Creates 2,000 samples for validation
- Copies images to `/storage/1/saima/oneDM/upti2_raw_nodeg/train/` and `val/`
- Creates `train.txt` and `val.txt` in format: `writer_id,image_name transcription`

**Verify the dataset:**
```bash
# Check that only nodegradation images are included
head -20 /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt
# You should see writer IDs like: "AlviNastaleeq_nodegradation_123"
# NOT: "AlviNastaleeq_high_123" or "AlviNastaleeq_low_123"
```

**Expected output:**
```
/storage/1/saima/oneDM/upti2_raw_nodeg/
├── train/
│   ├── AlviNastaleeq_nodegradation_1.png
│   ├── PakNastaleeq_nodegradation_2.png
│   └── ... (ALL nodegradation only)
├── val/
├── train.txt
└── val.txt
```

---

### Step 2: Preprocess Images (Resize and Create Laplace Edges)

```bash
python preprocess_upti2_images.py \
  --data_dir /storage/1/saima/oneDM/upti2_raw_nodeg \
  --output_dir /storage/1/saima/oneDM/upti2_processed_nodeg \
  --threads 16
```

**What this does:**
- Resizes all images to height=64 pixels (width scaled proportionally)
- Creates Laplace edge detection images for style conditioning
- Saves to `/storage/1/saima/oneDM/upti2_processed_nodeg/` and `upti2_processed_nodeg_laplace/`

**Expected output:**
```
/storage/1/saima/oneDM/upti2_processed_nodeg/
├── train/ (resized images)
└── val/

/storage/1/saima/oneDM/upti2_processed_nodeg_laplace/
├── train/ (edge maps)
└── val/
```

---

### Step 3: Copy Annotation Files

```bash
mkdir -p /storage/1/saima/oneDM/Arabic-One-DM/data
cp /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt /storage/1/saima/oneDM/Arabic-One-DM/data/
cp /storage/1/saima/oneDM/upti2_raw_nodeg/val.txt /storage/1/saima/oneDM/Arabic-One-DM/data/
```

---

### Step 4: Update Config File

Update `configs/UPTI64.yml` to use the new nodegradation dataset:

```yaml
DATA_LOADER:
  NUM_THREADS: 8
  IAMGE_PATH: /storage/1/saima/oneDM/upti2_processed_nodeg
  STYLE_PATH: /storage/1/saima/oneDM/upti2_processed_nodeg
  LAPLACE_PATH: /storage/1/saima/oneDM/upti2_processed_nodeg_laplace
OUTPUT_DIR: /storage/1/saima/oneDM/output_upti2_nodeg
```

---

### Step 5: Verify Setup Before Training 🔍

**Run these checks to ensure everything is correct:**

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

# 1. Check dataset files exist
ls -lh data/train.txt data/val.txt

# 2. Check processed images exist
ls /storage/1/saima/oneDM/upti2_processed_nodeg/train/ | head -10

# 3. Test content encoding (verify text → tensor works)
python3 << 'EOF'
from data_loader.loader_ara import ContentData
content_data = ContentData()
test_text = 'مخالفت کےباوجود'
content_tensor = content_data.get_content(test_text)
print(f"✓ Content tensor shape: {content_tensor.shape}")
print(f"✓ Non-zero values: {(content_tensor.abs() > 0.01).sum().item()}")
if (content_tensor.abs() > 0.01).sum() == 0:
    print("✗ ERROR: Content is all zeros!")
else:
    print("✓ Content encoding OK")
EOF
```

---

### Step 6: Start Training 🚀

**Single GPU training:**

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI64.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_nodeg_clean
```

**Multi-GPU training (if you have 4 GPUs):**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI64.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_nodeg_clean
```

---

### Step 7: Monitor Training 📊

**Watch training progress in terminal:**
```bash
# Training will print epoch progress and loss values
# Look for:
# - Epoch X/350
# - reconstruct_loss, high_nce_loss, low_nce_loss values
```

**View TensorBoard logs:**
```bash
# On server:
cd /storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean

# Start TensorBoard
tensorboard --logdir=tboard --port=6006 --bind_all

# Then on your local machine:
ssh -L 6006:localhost:6006 tukl@your_server_ip

# Open browser: http://localhost:6006
# You'll see graphs of all loss values over time
```

**Check validation samples:**
```bash
# Every epoch, the model generates sample images
ls -lht /storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean/sample/ | head -20

# View latest samples to see if model is learning
```

**Check model checkpoints:**
```bash
# Checkpoints are saved every 50 epochs
ls -lh /storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean/model/
# Should see: 49-ckpt.pt, 99-ckpt.pt, 149-ckpt.pt, etc.
```

---

## Expected Training Behavior

### Early epochs (1-20):
- Loss values will be high (~0.5-1.0 for reconstruct_loss)
- Validation samples will look random/messy
- **This is normal!** The model is still learning

### Mid training (20-100):
- Loss should gradually decrease
- Some recognizable Urdu letter shapes should appear
- Text should start aligning with input prompts

### Late training (100-200):
- Loss stabilizes around 0.1-0.3
- Clear, readable Urdu text in validation samples
- Text matches input prompts accurately

### Signs of problems:
- ✗ Loss stays flat (not decreasing)
- ✗ Loss explodes (goes to NaN or >10)
- ✗ Validation samples show no improvement after 50 epochs
- ✗ All validation text looks identical/random

---

## Debugging Training Issues

### Issue 1: Model generates wrong characters

**Check content encoding:**
```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

python3 << 'EOF'
from data_loader.loader_ara import ContentData, IAMDataset
import torch

# Test ContentData (used in validation)
content = ContentData()
test_texts = [
    'مخالفت کےباوجود',
    'بقاء اسی نظریہ',
    'میں مصروف ہوگیا'
]

for text in test_texts:
    tensor = content.get_content(text)
    non_zero = (tensor.abs() > 0.01).sum().item()
    print(f"Text: {text}")
    print(f"  Shape: {tensor.shape}, Non-zero: {non_zero}")
    if non_zero < len(text):
        print(f"  ✗ WARNING: Too few non-zero values!")
    else:
        print(f"  ✓ OK")
EOF
```

### Issue 2: Loss not decreasing

**Possible causes:**
1. Learning rate too low/high
2. Batch size too small
3. Dataset corruption
4. Model not using content conditioning

**Check learning rate:**
```bash
grep "BASE_LR" configs/UPTI64.yml
# Should be: 0.0001 (good default)
```

### Issue 3: Out of memory

**Reduce batch size:**
```bash
nano configs/UPTI64.yml
# Change: IMS_PER_BATCH: 16  (or 8 for very limited GPU)
```

---

## Important Configuration Parameters

### In `configs/UPTI64.yml`:

```yaml
SOLVER:
  BASE_LR: 0.0001          # Learning rate
  EPOCHS: 350              # Total epochs
  WARMUP_ITERS: 25000      # Warmup steps
  GRAD_L2_CLIP: 5.0        # Gradient clipping

TRAIN:
  IMS_PER_BATCH: 16        # Batch size (adjust for GPU memory)
  SNAPSHOT_BEGIN: 1        # Start saving checkpoints
  SNAPSHOT_ITERS: 50       # Save every 50 epochs
  VALIDATE_BEGIN: 0        # Start validation
  VALIDATE_ITERS: 1        # Validate every epoch
  IMG_H: 64                # Image height
  IMG_W: 1024              # Max image width

MODEL:
  EMB_DIM: 512             # Model dimension
  NUM_HEADS: 4             # Attention heads
  NUM_RES_BLOCKS: 1        # Residual blocks
```

---

## Testing the Trained Model

After training completes (or reaches ~100+ epochs):

```bash
cd /storage/1/saima/oneDM/Arabic-One-DM

python test.py \
  --cfg configs/UPTI64.yml \
  --one_dm /storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean/model/149-ckpt.pt \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --save_dir /storage/1/saima/oneDM/test_results_nodeg \
  --generate_type oov_u
```

---

## Summary of All Commands (Quick Reference)

```bash
# === SETUP (Run once) ===
cd /storage/1/saima/oneDM/Arabic-One-DM

# 1. Prepare dataset (nodegradation only)
python prepare_upti2_dataset.py \
  --image_base /storage/1/saima/images_upti2_2/images \
  --gt_base /home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth \
  --output_base /storage/1/saima/oneDM/upti2_raw_nodeg \
  --max_train 20000 --max_val 2000 --seed 42

# 2. Preprocess images
python preprocess_upti2_images.py \
  --data_dir /storage/1/saima/oneDM/upti2_raw_nodeg \
  --output_dir /storage/1/saima/oneDM/upti2_processed_nodeg \
  --threads 16

# 3. Copy annotations
mkdir -p data
cp /storage/1/saima/oneDM/upti2_raw_nodeg/train.txt data/
cp /storage/1/saima/oneDM/upti2_raw_nodeg/val.txt data/

# 4. Update configs/UPTI64.yml paths to use upti2_processed_nodeg

# === TRAINING ===
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29500 \
  train.py \
  --cfg configs/UPTI64.yml \
  --stable_dif_path runwayml/stable-diffusion-v1-5 \
  --log upti2_nodeg_clean

# === MONITORING ===
# View samples:
ls -lht /storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean/sample/ | head

# View TensorBoard:
tensorboard --logdir=/storage/1/saima/oneDM/output_upti2_nodeg/UPTI64/upti2_nodeg_clean/tboard --port=6006
```

---

## Expected Training Time

- **With 1 GPU (RTX 3090)**: ~2-3 days for 200 epochs, ~4-5 days for 350 epochs
- **With 4 GPUs**: ~12-18 hours for 200 epochs, ~24-30 hours for 350 epochs
- **Dataset size**: 20,000 nodegradation samples

---

## Storage Requirements

- **Dataset (raw)**: ~2GB
- **Dataset (processed)**: ~2GB
- **Checkpoints** (3 kept): ~6GB
- **Sample images**: ~100MB
- **TensorBoard logs**: ~50MB
- **Total needed**: ~10.2GB

---

## Troubleshooting Checklist

If training isn't working:

- [ ] Dataset uses ONLY nodegradation images (check train.txt)
- [ ] Config file paths point to `upti2_processed_nodeg`
- [ ] ContentData text encoding produces non-zero tensors
- [ ] GPU has enough memory (check with `nvidia-smi`)
- [ ] Batch size fits in GPU memory (reduce if OOM)
- [ ] Training loss is decreasing (check TensorBoard)
- [ ] Validation samples show gradual improvement
- [ ] No error messages in terminal output

---

## Questions or Issues?

If you encounter problems:
1. Check terminal output for error messages
2. View TensorBoard to see if loss is decreasing
3. Check validation samples for visual improvements
4. Verify dataset preparation steps completed correctly
5. Compare your config file with the guide

**Common issues:**
- Out of memory → Reduce batch size
- No improvement after 50 epochs → Check content encoding
- Loss explodes (NaN) → Reduce learning rate or check data
- Wrong characters → Re-run dataset preparation with fixes

---

## Changelog

### Feb 10, 2026
- ✅ Fixed: Dataset now filters for nodegradation images only
- ✅ Fixed: ContentData letter2index mapping for Urdu text
- ✅ Added: Verification steps before training
- ✅ Added: TensorBoard monitoring instructions
- ✅ Added: Debugging section for common issues
- ✅ Updated: All paths to use `upti2_processed_nodeg`
