# Training Updates for Arabic One-DM

## 🔍 Root Cause of Text Cutoff Issue

**Problem**: Generated Urdu text appears cut off or extends beyond image boundaries.

**Root Cause**: The model was generating images with **fixed width based on style reference images**, not the actual **text content length**. For long Urdu sentences, this caused text to be truncated.

**Previous Code** (trainer.py:220-221):
```python
latent_width = single_style.shape[3]//8  # Fixed width from style image
x = torch.randn((1, 4, single_style.shape[2]//8, latent_width)).to(self.device)
```

**Fixed Code**: Now calculates width dynamically based on text length:
```python
text_length = len(text)
estimated_width = min(text_length * 2 + 4, 128)  # Adaptive width, capped at 1024px
latent_width = estimated_width
```

---

## ✅ Changes Made

### 1. **Fixed Text Generation Width** (trainer.py)
   - **Before**: Used fixed style image width → text cutoff
   - **After**: Calculates width based on text length → proper fitting
   - Formula: `width = min(text_length * 2 + 4, 128)` latent units (max 1024px)

### 2. **Model Saving Updates** (configs/UPTI64.yml + trainer.py)
   - **Checkpoint saving**: Every **20 epochs** (changed from 50)
   - **Best model**: Automatically saved as `best_model.pt` based on lowest training loss
   - **Cleanup**: Only keeps last 3 checkpoints to save disk space

### 3. **Validation Texts** (trainer.py)
   Updated to your 2 specified lines:
   - 'بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔'
   - 'لیکن بدقسمتی سےپاکستان بننے کے بعد ہی اس کے اندر ایسے دشمن'

### 4. **Training Epochs** (configs/UPTI64.yml)
   - **Before**: 350 epochs
   - **After**: 1000 epochs (matches official One-DM)

---

## 📊 Architecture Comparison with Official One-DM

| Parameter | Official One-DM | Your Setup | Match? |
|-----------|----------------|------------|--------|
| **Epochs** | 1000 | ~~350~~ → **1000** | ✅ Fixed |
| **Image Size** | 64×64 (words) | 64×1024 (sentences) | ✅ Correct for Urdu sentences |
| **Model Channels** | 512 | 512 | ✅ |
| **Attention Heads** | 4 | 4 | ✅ |
| **Res Blocks** | 1 | 1 | ✅ |
| **In/Out Channels** | 4 | 4 | ✅ |

**Note**: Your wide aspect ratio (64×1024) is **correct** for full Urdu sentences, unlike the official repo which trains on single English words (64×64).

---

## 🚀 Next Steps

1. **Pull changes to server**:
   ```bash
   cd /path/to/Arabic-One-DM
   git pull
   ```

2. **Re-prepare data** (if needed):
   - Ensure all training images are exactly **64×1024** pixels
   - Check that the previous 1034→1024 fix was applied

3. **Resume training**:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py \
       --cfg configs/UPTI64.yml \
       --stable_dif_path runwayml/stable-diffusion-v1-5
   ```

4. **Monitor checkpoints**:
   - Regular checkpoints: `{epoch}-ckpt.pt` (every 20 epochs)
   - Best model: `best_model.pt` (automatically saved)
   - Validation images: `epoch{N}_val1.png`, `epoch{N}_val2.png`

---

## 📝 Files Modified

1. **configs/UPTI64.yml**
   - ✅ EPOCHS: 350 → 1000
   - ✅ SNAPSHOT_ITERS: 50 → 20

2. **trainer/trainer.py**
   - ✅ Added best model tracking (`self.best_loss`, `self.best_epoch`)
   - ✅ Fixed generation width calculation (line ~217-222)
   - ✅ Updated validation texts to 2 lines (line ~203-206)
   - ✅ Modified `_save_checkpoint()` to save best model
   - ✅ Updated checkpoint saving logic to pass average loss

---

## 💡 Tips

- **Text too long?** Adjust the width cap: `estimated_width = min(text_length * 2 + 4, 128)`
  - Current cap: 128 latent units = 1024 pixels
  - Increase if needed: change `128` to higher value

- **Memory issues?** Reduce batch size in config:
  ```yaml
  TRAIN:
    IMS_PER_BATCH: 16  # Reduce to 8 or 12 if needed
  ```

- **Check progress**: Look for these log messages during training:
  ```
  [Checkpoint] Saving model at epoch X
  ✓ Best model saved! (epoch X, loss: Y)
  [Validation] Generating sample images for epoch X
  ```
