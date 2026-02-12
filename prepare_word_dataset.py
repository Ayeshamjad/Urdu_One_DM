"""
Preprocessing script for word-level Urdu handwritten dataset
Processes images and creates the required file structure for One-DM training

Dataset structure expected:
- {dataset_root}/train/images/
- {dataset_root}/train/train_gt.txt
- {dataset_root}/val/images/
- {dataset_root}/val/val_gt.txt
- {dataset_root}/test/images/
- {dataset_root}/test/test_gt.txt

GT file format: images/1.jpg\tword (tab-separated)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from tqdm import tqdm
import argparse

# Target image dimensions for word-level data
IMG_HEIGHT = 64
IMG_WIDTH = 256  # Smaller than sentence-level (words are shorter)


def resize_and_pad(image, target_h=IMG_HEIGHT, target_w=IMG_WIDTH):
    """
    Resize image to target height while maintaining aspect ratio,
    then pad/crop to target width.
    """
    img_h, img_w = image.shape[:2]

    # Resize to target height keeping aspect ratio
    new_w = int(img_w * target_h / img_h)
    image = cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Handle width
    if new_w > target_w:
        # Crop if too wide
        image = image[:, :target_w]
    elif new_w < target_w:
        # Pad if too narrow (pad on right side)
        pad_w = target_w - new_w
        image = cv2.copyMakeBorder(image, 0, 0, 0, pad_w,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return image


def compute_laplace(image):
    """Compute Laplace edge detection for style representation."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Apply Laplace filter
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = np.abs(laplace)
    laplace = np.clip(laplace, 0, 255).astype(np.uint8)

    return laplace


def process_split(dataset_root, split='train', output_base='/storage/1/saima/oneDM/word_processed'):
    """
    Process one split (train/val/test) of the dataset.

    Creates:
    - {output_base}/{split}/images/          - processed RGB images
    - {output_base}/{split}/laplace/         - Laplace edge images
    - {output_base}/{split}/gt/              - individual text files (one per image)
    - {output_base}/{split}/{split}_gt.txt   - copied ground truth file
    """

    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}")

    # Input paths
    input_img_dir = os.path.join(dataset_root, split, 'images')
    input_gt_file = os.path.join(dataset_root, split, f'{split}_gt.txt')

    # Output paths
    output_split_dir = os.path.join(output_base, split)
    output_img_dir = os.path.join(output_split_dir, 'images')
    output_laplace_dir = os.path.join(output_split_dir, 'laplace')
    output_gt_dir = os.path.join(output_split_dir, 'gt')

    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_laplace_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)

    # Check input files exist
    if not os.path.exists(input_img_dir):
        print(f"ERROR: Input image directory not found: {input_img_dir}")
        return

    if not os.path.exists(input_gt_file):
        print(f"ERROR: Ground truth file not found: {input_gt_file}")
        return

    # Read ground truth file
    print(f"Reading ground truth from: {input_gt_file}")
    with open(input_gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} entries in ground truth file")

    # Process each image
    processed = 0
    skipped = 0

    for line in tqdm(lines, desc=f"Processing {split}"):
        line = line.strip()
        if not line:
            continue

        # Parse line: images/1.jpg\tword
        if '\t' in line:
            img_rel_path, text = line.split('\t', 1)
        else:
            # Try space separation as fallback
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}")
                skipped += 1
                continue
            img_rel_path, text = parts

        # Remove "images/" prefix if present
        if img_rel_path.startswith('images/'):
            img_name = img_rel_path[len('images/'):]
        else:
            img_name = img_rel_path

        # Build full input path
        input_img_path = os.path.join(input_img_dir, img_name)

        if not os.path.exists(input_img_path):
            print(f"Warning: Image not found: {input_img_path}")
            skipped += 1
            continue

        try:
            # Read image
            image = cv2.imread(input_img_path)
            if image is None:
                print(f"Warning: Failed to read image: {input_img_path}")
                skipped += 1
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize and pad
            processed_img = resize_and_pad(image, IMG_HEIGHT, IMG_WIDTH)

            # Compute Laplace
            laplace_img = compute_laplace(processed_img)

            # Save processed image (RGB)
            output_img_path = os.path.join(output_img_dir, img_name)
            cv2.imwrite(output_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

            # Save Laplace image
            output_laplace_path = os.path.join(output_laplace_dir, img_name)
            cv2.imwrite(output_laplace_path, laplace_img)

            # Save ground truth text file
            img_base = os.path.splitext(img_name)[0]
            gt_txt_path = os.path.join(output_gt_dir, f"{img_base}.txt")
            with open(gt_txt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            processed += 1

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            skipped += 1
            continue

    print(f"\n{split} split complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output directory: {output_split_dir}")

    # Copy the ground truth file to output directory
    output_gt_file = os.path.join(output_split_dir, f'{split}_gt.txt')
    with open(output_gt_file, 'w', encoding='utf-8') as out_f:
        out_f.writelines(lines)
    print(f"  Ground truth file copied to: {output_gt_file}")

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(description='Preprocess word-level Urdu dataset for One-DM')
    parser.add_argument('--input', type=str,
                        default='/storage/1/saima/oneDM/transfer_folder',
                        help='Root directory of input dataset')
    parser.add_argument('--output', type=str,
                        default='/storage/1/saima/oneDM/word_processed',
                        help='Output directory for processed dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to process')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Word-level Urdu Dataset Preprocessing for One-DM")
    print(f"{'='*60}")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Target image size: {IMG_HEIGHT}x{IMG_WIDTH} (H x W)")
    print(f"Splits to process: {', '.join(args.splits)}")

    total_processed = 0
    total_skipped = 0

    for split in args.splits:
        processed, skipped = process_split(args.input, split, args.output)
        total_processed += processed
        total_skipped += skipped

    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"\nProcessed dataset location: {args.output}")
    print("\nNext steps:")
    print("1. Update config file DATA_LOADER paths to point to processed dataset")
    print("2. Run training: python -m torch.distributed.launch --nproc_per_node=N train.py --cfg configs/UPTI_Word64.yml")


if __name__ == '__main__':
    main()
