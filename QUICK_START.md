# Quick Start Guide: Word-Level Urdu One-DM Training

## 🚀 3-Step Setup

### Step 1: Preprocess Dataset (5-10 minutes)
```bash
cd /home/osama/Desktop/Ayesha/oneDM/Arabic-One-DM

python prepare_word_dataset.py \
    --input /storage/1/saima/oneDM/transfer_folder \
    --output /storage/1/saima/oneDM/word_processed \
    --splits train val test
```

**What it does**: Resizes images to 64×256, generates Laplace edges, creates structured dataset

**Output**: `/storage/1/saima/oneDM/word_processed/{train,val,test}/{images,laplace,gt}/`

---

### Step 2: Start Training (Multi-GPU Recommended)
```bash
# For 2 GPUs:
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --feat_model <path_to_resnet18.pth> \
    --log word_training

# For 4 GPUs:
python -m torch.distributed.launch --nproc_per_node=4 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --log word_training
```

**Note**: `--feat_model` is optional (for pretrained style encoder)

---

### Step 3: Monitor Progress
```bash
# View logs
tail -f /storage/1/saima/oneDM/output_word_urdu/logs/word_training.log

# TensorBoard
tensorboard --logdir /storage/1/saima/oneDM/output_word_urdu/tboard

# Check validation images
ls /storage/1/saima/oneDM/output_word_urdu/samples/

# Check checkpoints
ls /storage/1/saima/oneDM/output_word_urdu/models/
```

---

## 🔄 Resume Training (If Interrupted)
```bash
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --one_dm /storage/1/saima/oneDM/output_word_urdu/models/199-ckpt.pt \
    --log word_training_resumed
```

---

## 📋 What Changed?

### New Files:
1. **`prepare_word_dataset.py`** - Preprocessing script
2. **`configs/UPTI_Word64.yml`** - Word-level config
3. **`WORD_DATASET_SETUP.md`** - Full documentation
4. **`CHANGES_SUMMARY.md`** - Detailed change log

### Modified Files:
1. **`data_loader/loader_ara.py`** - Added word-level format support
2. **`trainer/trainer.py`** - Updated validation texts to words

### Architecture:
- ✅ **Official One-DM**: EMB_DIM=512, NUM_HEADS=4, NUM_RES_BLOCKS=1
- ✅ **Checkpoint saving**: Every 20 epochs + best model
- ✅ **Auto-cleanup**: Keeps last 3 checkpoints

---

## 📊 Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Image Size | 64×256 | Optimized for words (vs 1024 for sentences) |
| Batch Size | 16 | Increased (words are smaller) |
| Epochs | 1000 | Official One-DM standard |
| Save Frequency | Every 20 | Resume-friendly |
| Learning Rate | 0.0001 | Official One-DM |

---

## 📁 Dataset Structure

**Input** (your raw dataset):
```
/storage/1/saima/oneDM/transfer_folder/
├── train/
│   ├── images/          # Raw word images
│   └── train_gt.txt     # images/1.jpg\tword
├── val/
│   ├── images/
│   └── val_gt.txt
└── test/
    ├── images/
    └── test_gt.txt
```

**Output** (after preprocessing):
```
/storage/1/saima/oneDM/word_processed/
├── train/
│   ├── images/          # Processed 64×256 images
│   ├── laplace/         # Edge images
│   └── gt/              # Individual text files
├── val/
└── test/
```

**Training Output**:
```
/storage/1/saima/oneDM/output_word_urdu/
├── models/
│   ├── 19-ckpt.pt      # Epoch 20
│   ├── 39-ckpt.pt      # Epoch 40
│   ├── 59-ckpt.pt      # Epoch 60 (only last 3)
│   └── best_model.pt   # Best model
├── samples/            # Validation images
├── tboard/             # TensorBoard logs
└── logs/               # Training logs
```

---

## ⏱️ Expected Timeline

- **Preprocessing**: 5-10 minutes (~10K images)
- **First epoch**: ~2-5 minutes (depends on GPU/dataset)
- **First checkpoint**: After epoch 20 (~1 hour)
- **Visible results**: Around epoch 100-200
- **Full training**: 1000 epochs (~3-5 days on 2 GPUs)

---

## ✅ Verification Commands

```bash
# 1. Check preprocessing output
ls /storage/1/saima/oneDM/word_processed/train/images/ | wc -l

# 2. Verify image dimensions
python -c "
import cv2
img = cv2.imread('/storage/1/saima/oneDM/word_processed/train/images/1.jpg')
print(f'Image shape: {img.shape}')  # Should be (64, 256, 3)
"

# 3. Count training samples
grep -c "" /storage/1/saima/oneDM/word_processed/train/train_gt.txt

# 4. Check GPU usage during training
nvidia-smi

# 5. Verify checkpoint saving
ls -lth /storage/1/saima/oneDM/output_word_urdu/models/
```

---

## 🔧 Common Issues

### Issue: "Image not found" during preprocessing
```bash
# Check input structure
ls /storage/1/saima/oneDM/transfer_folder/train/images/ | head
cat /storage/1/saima/oneDM/transfer_folder/train/train_gt.txt | head
```

### Issue: CUDA out of memory
Edit `configs/UPTI_Word64.yml`:
```yaml
TRAIN:
  IMS_PER_BATCH: 8  # Reduce from 16
```

### Issue: "No style images found"
```bash
# Verify preprocessing completed
ls /storage/1/saima/oneDM/word_processed/train/images/ | wc -l
ls /storage/1/saima/oneDM/word_processed/train/laplace/ | wc -l
```

---

## 📚 Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| **`QUICK_START.md`** | Quick reference (this file) | First time setup |
| **`WORD_DATASET_SETUP.md`** | Detailed guide | Troubleshooting |
| **`CHANGES_SUMMARY.md`** | Technical changes | Understanding code |
| **`configs/UPTI_Word64.yml`** | Configuration | Tuning parameters |

---

## 🎯 Success Checklist

- [ ] Raw dataset at `/storage/1/saima/oneDM/transfer_folder/`
- [ ] Run `prepare_word_dataset.py` successfully
- [ ] Processed dataset at `/storage/1/saima/oneDM/word_processed/`
- [ ] Start training with multi-GPU command
- [ ] See checkpoint saved at epoch 20
- [ ] Validation images generated every epoch
- [ ] Monitor TensorBoard for decreasing loss
- [ ] Best model updates when loss improves

---

## 💡 Pro Tips

1. **Test preprocessing on small subset first**:
   ```bash
   python prepare_word_dataset.py \
       --input /storage/1/saima/oneDM/transfer_folder \
       --output /tmp/word_test \
       --splits train
   ```

2. **Start with fewer epochs to test**:
   Edit `configs/UPTI_Word64.yml`:
   ```yaml
   SOLVER:
     EPOCHS: 100  # Test with 100 first
   ```

3. **Monitor GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Check validation images regularly**:
   ```bash
   eog /storage/1/saima/oneDM/output_word_urdu/samples/epoch*_val*.png
   ```

---

## 🆘 Need Help?

1. **Check logs**: `/storage/1/saima/oneDM/output_word_urdu/logs/word_training.log`
2. **Read detailed guide**: `WORD_DATASET_SETUP.md`
3. **Review changes**: `CHANGES_SUMMARY.md`
4. **Verify config**: `configs/UPTI_Word64.yml`

---

## 📞 Summary

**3 commands to train word-level Urdu One-DM:**

1. Preprocess: `python prepare_word_dataset.py --input ... --output ...`
2. Train: `python -m torch.distributed.launch --nproc_per_node=2 train.py --cfg configs/UPTI_Word64.yml ...`
3. Monitor: `tail -f /storage/1/saima/oneDM/output_word_urdu/logs/word_training.log`

**Architecture**: Official One-DM (512 channels, 4 heads, 1 res block)
**Checkpoints**: Every 20 epochs + best model
**Ready to train**: Just run the commands above! 🚀
