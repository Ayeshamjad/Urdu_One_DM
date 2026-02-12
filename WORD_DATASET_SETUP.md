# Word-Level Urdu Handwritten Dataset Setup for One-DM

## Overview
This guide explains how to train One-DM on word-level Urdu handwritten dataset (shifted from sentence-level UPTI dataset).

## Dataset Structure
Your dataset should follow this structure:
```
/storage/1/saima/oneDM/transfer_folder/
├── train/
│   ├── images/          # Training word images
│   ├── train_gt.txt     # Ground truth: images/1.jpg\tword
│   └── vocabulary.txt   # Optional: list of unique words
├── val/
│   ├── images/          # Validation word images
│   └── val_gt.txt       # Ground truth
└── test/
    ├── images/          # Test word images
    └── test_gt.txt      # Ground truth
```

## Changes Made

### 1. **New Files Created**

#### `prepare_word_dataset.py`
- **Purpose**: Preprocesses word-level images for training
- **What it does**:
  - Resizes all images to 64×256 (H×W) - optimized for words
  - Pads/crops images to maintain consistent dimensions
  - Generates Laplace edge images for style representation
  - Creates individual text files for each image
  - Organizes data into One-DM compatible structure

#### `configs/UPTI_Word64.yml`
- **Purpose**: Configuration file for word-level training
- **Key settings**:
  - Image size: 64×256 (smaller width for words vs 1024 for sentences)
  - Batch size: 16 (increased since words are smaller)
  - Architecture: Matches official One-DM (512 channels, 4 heads, 1 res block)
  - Epochs: 1000 (official One-DM standard)
  - Checkpoint saving: Every 20 epochs + best model tracking
  - Dataset format: 'word'

### 2. **Modified Files**

#### `data_loader/loader_ara.py`
- **Added**: `load_data_word()` method
  - Loads word-level dataset from `{split}_gt.txt` files
  - Handles tab-separated format: `images/1.jpg\tword`
  - All words share single "writer" ID for style sampling

- **Updated**: `get_style_ref()` method
  - Added word-level format handling
  - Randomly samples style images from all training images
  - Generates Laplace on-the-fly if not pre-computed

- **Updated**: `__getitem__()` method
  - Added path handling for word-level format

#### `trainer/trainer.py`
- **Updated**: Validation texts
  - Changed from full Urdu sentences to individual words
  - Uses vocabulary from training data: ['جمہویریہ', 'تابوت', 'بحری', ...]
  - Better suited for word-level generation

### 3. **Preserved Features** (Already Implemented)

#### Model Architecture
- ✅ Matches official One-DM architecture:
  - Model channels: 512
  - Attention heads: 4
  - Residual blocks: 1
  - Context dimension: 512

#### Checkpoint Saving (trainer.py)
- ✅ Saves checkpoint every 20 epochs (configurable via `SNAPSHOT_ITERS`)
- ✅ Tracks best model based on validation loss
- ✅ Auto-deletes old checkpoints (keeps last 3)
- ✅ Saves two types of models:
  1. `{epoch}-ckpt.pt` - Regular checkpoints every 20 epochs
  2. `best_model.pt` - Best model based on lowest loss

## Step-by-Step Usage

### Step 1: Preprocess the Dataset

Run the preprocessing script to prepare images:

```bash
cd /home/osama/Desktop/Ayesha/oneDM/Arabic-One-DM

# Preprocess all splits (train, val, test)
python prepare_word_dataset.py \
    --input /storage/1/saima/oneDM/transfer_folder \
    --output /storage/1/saima/oneDM/word_processed \
    --splits train val test
```

**What this does**:
- Reads raw images from `transfer_folder/train/images/`, `val/images/`, `test/images/`
- Resizes to 64×256 pixels
- Generates Laplace edge images
- Creates processed dataset at `/storage/1/saima/oneDM/word_processed/`

**Output structure**:
```
/storage/1/saima/oneDM/word_processed/
├── train/
│   ├── images/      # Processed 64×256 RGB images
│   ├── laplace/     # Laplace edge images
│   ├── gt/          # Individual .txt files per image
│   └── train_gt.txt # Copy of original ground truth
├── val/
│   ├── images/
│   ├── laplace/
│   ├── gt/
│   └── val_gt.txt
└── test/
    ├── images/
    ├── laplace/
    ├── gt/
    └── test_gt.txt
```

### Step 2: Verify Preprocessing

Check the output:
```bash
# Count processed images
ls /storage/1/saima/oneDM/word_processed/train/images/ | wc -l
ls /storage/1/saima/oneDM/word_processed/val/images/ | wc -l

# Verify image dimensions
python -c "
import cv2
img = cv2.imread('/storage/1/saima/oneDM/word_processed/train/images/1.jpg')
print(f'Image shape: {img.shape}')  # Should be (64, 256, 3)
"
```

### Step 3: Update Config (if needed)

The config file `configs/UPTI_Word64.yml` is already set up with correct paths. Verify:

```yaml
DATA_LOADER:
  DATASET_FORMAT: word
  IAMGE_PATH: /storage/1/saima/oneDM/word_processed
  STYLE_PATH: /storage/1/saima/oneDM/word_processed
  LAPLACE_PATH: /storage/1/saima/oneDM/word_processed

OUTPUT_DIR: /storage/1/saima/oneDM/output_word_urdu
```

### Step 4: Train the Model

#### Single GPU Training:
```bash
cd /home/osama/Desktop/Ayesha/oneDM/Arabic-One-DM

python train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --feat_model <path_to_pretrained_resnet18.pth> \
    --log word_training
```

#### Multi-GPU Training (Recommended):
```bash
# For 2 GPUs:
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --feat_model <path_to_pretrained_resnet18.pth> \
    --log word_training

# For 4 GPUs:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --log word_training
```

### Step 5: Resume Training (if interrupted)

To resume from a checkpoint:
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --one_dm /storage/1/saima/oneDM/output_word_urdu/models/199-ckpt.pt \
    --log word_training_resumed
```

### Step 6: Monitor Training

#### Check Logs:
```bash
# View training progress
tail -f /storage/1/saima/oneDM/output_word_urdu/logs/word_training.log

# Monitor TensorBoard
tensorboard --logdir /storage/1/saima/oneDM/output_word_urdu/tboard
```

#### Validation Images:
Generated validation images are saved at:
```
/storage/1/saima/oneDM/output_word_urdu/samples/
├── epoch1_val1.png    # Word: جمہویریہ
├── epoch1_val2.png    # Word: تابوت
├── epoch1_val3.png    # Word: بحری
...
```

#### Checkpoints:
Models are saved at:
```
/storage/1/saima/oneDM/output_word_urdu/models/
├── 19-ckpt.pt         # Epoch 20
├── 39-ckpt.pt         # Epoch 40
├── 59-ckpt.pt         # Epoch 60 (only last 3 kept)
└── best_model.pt      # Best model based on validation loss
```

## Training Configuration Details

### Architecture (Official One-DM)
- **Model channels**: 512
- **Attention heads**: 4
- **Residual blocks**: 1
- **Context dimension**: 512
- **Input/Output channels**: 4 (latent space)

### Training Settings
- **Learning rate**: 0.0001
- **Optimizer**: AdamW
- **Gradient clipping**: 5.0
- **Total epochs**: 1000
- **Batch size**: 16 (train), 8 (val)
- **Image size**: 64×256 (H×W)

### Checkpoint Strategy
- **Save frequency**: Every 20 epochs
- **Validation frequency**: Every epoch
- **Best model tracking**: Automatically saves model with lowest loss
- **Auto-cleanup**: Keeps only last 3 checkpoints (saves disk space)

## Key Differences: Sentence vs Word Level

| Aspect | Sentence-Level (UPTI) | Word-Level (New) |
|--------|----------------------|------------------|
| Image width | 1024 pixels | 256 pixels |
| Batch size | 8 | 16 |
| Training samples | ~10K sentences | Variable (depends on dataset) |
| Validation texts | Full sentences | Individual words |
| Style sampling | Per writer ID | Random from all images |
| Dataset format | UPTI structure | Simple GT files |

## Troubleshooting

### Issue: "Image not found" during preprocessing
**Solution**: Check that input directory has correct structure:
```bash
ls /storage/1/saima/oneDM/transfer_folder/train/images/
cat /storage/1/saima/oneDM/transfer_folder/train/train_gt.txt | head
```

### Issue: Out of memory during training
**Solutions**:
1. Reduce batch size in config: `TRAIN.IMS_PER_BATCH: 8`
2. Use gradient accumulation
3. Use fewer GPUs with larger per-GPU batch

### Issue: "No style images found"
**Solution**: Verify preprocessing created images:
```bash
ls /storage/1/saima/oneDM/word_processed/train/images/ | head
ls /storage/1/saima/oneDM/word_processed/train/laplace/ | head
```

### Issue: Training loss not decreasing
**Checks**:
1. Verify dataset has sufficient samples (>1000 words)
2. Check validation images show improvement
3. Consider loading pretrained model: `--one_dm <pretrained_ckpt.pt>`

## Expected Training Timeline

- **Preprocessing**: 5-10 minutes (for ~10K images)
- **Epoch time**: ~2-5 minutes per epoch (depends on dataset size, GPU)
- **First checkpoint**: After epoch 20 (~1 hour)
- **Visible results**: Around epoch 100-200
- **Full training**: 1000 epochs (~3-5 days on 2 GPUs)

## Next Steps After Training

1. **Evaluate generated samples**:
   - Check validation images in `output_word_urdu/samples/`
   - Verify style transfer quality

2. **Test on custom words**:
   - Use `test.py` to generate specific words
   - Test with unseen vocabulary

3. **Fine-tune if needed**:
   - Resume from best model
   - Adjust learning rate or batch size

## References

- **Official One-DM**: Architecture and training settings match the official implementation
- **Dataset format**: Simplified from UPTI to standard GT file format
- **Model saving**: Enhanced with best model tracking and auto-cleanup

## Contact & Support

For issues or questions:
1. Check logs: `/storage/1/saima/oneDM/output_word_urdu/logs/`
2. Verify config: `configs/UPTI_Word64.yml`
3. Test preprocessing: `python prepare_word_dataset.py --help`
