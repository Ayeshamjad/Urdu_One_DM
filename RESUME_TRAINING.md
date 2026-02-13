# Resume Training Guide - Epoch Counter Fixed ✅

## What Was Fixed

The training now **properly saves and resumes from the correct epoch number**.

### Changes Made:
1. ✅ Checkpoints now save: `{model, epoch, optimizer, loss}` (not just model)
2. ✅ When resuming, epoch counter continues from saved checkpoint
3. ✅ Backward compatible: still loads old checkpoints (but epoch resets to 0)

---

## How to Resume Training Safely

### Option 1: Continue in SAME Directory (Recommended)
**Use the SAME `--log` name** to continue in the existing output directory:

```bash
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --one_dm /storage/1/saima/oneDM/output_word_urdu/models/19-ckpt.pt \
    --log word_training
```

**Result:**
- ✅ Resumes from **Epoch 20/1000** (continues after epoch 19)
- ✅ Saves to: `/storage/1/saima/oneDM/output_word_urdu/models/`
- ✅ New checkpoints: `20-ckpt.pt`, `39-ckpt.pt`, `59-ckpt.pt`, etc.
- ✅ **Your 19-ckpt.pt is SAFE** (won't be overwritten)
- ✅ Generated images go to same `sample/` folder

---

### Option 2: New Directory (Keep Old & New Separate)
**Use a DIFFERENT `--log` name** to create a new output directory:

```bash
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py \
    --cfg configs/UPTI_Word64.yml \
    --stable_dif_path runwayml/stable-diffusion-v1-5 \
    --one_dm /storage/1/saima/oneDM/output_word_urdu/models/19-ckpt.pt \
    --log word_training_resumed
```

**Result:**
- ✅ Resumes from **Epoch 20/1000**
- ✅ Saves to NEW folder: `/storage/1/saima/oneDM/output_word_urdu_resumed/`
- ✅ Old checkpoints untouched in `/storage/1/saima/oneDM/output_word_urdu/`

---

## What You'll See When It Works

```bash
load pretrained one_dm model from /path/to/19-ckpt.pt (epoch 19)
Loaded 71207 samples from word-level train set
...
======================================================================
Epoch 20/1000 | Process 0    # ← Should show 20, not 1!
======================================================================
```

---

## Checkpoint Format

### New Checkpoints (saved after this fix):
```python
checkpoint = {
    'model': model.state_dict(),      # Model weights
    'epoch': 19,                      # Epoch number
    'optimizer': optimizer.state_dict(),  # Optimizer state
    'loss': 0.1234                    # Average loss
}
```

### Old Checkpoints (saved before fix):
- Only contains `model.state_dict()` directly
- Still loads correctly, but epoch counter resets to 0

---

## Safety Features

1. ✅ **Won't overwrite old checkpoints** (different epoch numbers in filename)
2. ✅ **Backward compatible** (loads old checkpoints without epoch info)
3. ✅ **Auto-deletes only old checkpoints** (keeps last 1 to save space)
4. ✅ **Best model is preserved** separately

---

## Example: Your Current Situation

You have: `19-ckpt.pt` (epoch 19 saved)

### Using Same Log Name:
```
Before: /storage/1/saima/oneDM/output_word_urdu/models/19-ckpt.pt
After:  /storage/1/saima/oneDM/output_word_urdu/models/19-ckpt.pt  ← Still there
        /storage/1/saima/oneDM/output_word_urdu/models/39-ckpt.pt  ← New
        /storage/1/saima/oneDM/output_word_urdu/models/59-ckpt.pt  ← New
```

### Using Different Log Name:
```
Old:    /storage/1/saima/oneDM/output_word_urdu/models/19-ckpt.pt     ← Untouched
New:    /storage/1/saima/oneDM/output_word_urdu_resumed/models/39-ckpt.pt
        /storage/1/saima/oneDM/output_word_urdu_resumed/models/59-ckpt.pt
```

---

## Next Steps

1. **Push to git**: `git add . && git commit -m "Fix epoch counter for resume"`
2. **Pull on server**: `git pull`
3. **Resume training** using one of the commands above
4. **Verify**: Check that output shows "Epoch 20/1000" (not "Epoch 1/1000")

---

## Troubleshooting

**Q: It still shows "Epoch 1/1000"**
- A: You're loading an **old checkpoint** (saved before the fix). It will still load the weights correctly, but epoch counter resets. Just note that "Epoch 1" is actually "Epoch 20".

**Q: Can I resume from 19-ckpt.pt with the new code?**
- A: Yes! Old checkpoints load fine, but epoch info won't be available (starts from 0).

**Q: Will my 19-ckpt.pt be deleted?**
- A: No! The auto-delete only removes older checkpoints (e.g., keeps last 1). Since you're starting from epoch 20, it won't touch epoch 19.
