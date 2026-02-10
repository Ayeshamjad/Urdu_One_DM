#!/usr/bin/env python3
"""
Prepare UPTI2 dataset for oneDM training (Urdu handwriting)
This script:
1. Maps UPTI2 images to ground truth files
2. Creates train.txt and val.txt in the format: writer_id,image_name.png transcription
3. Limits dataset to specified number of samples
4. Organizes images in the expected directory structure
"""

import os
import shutil
from pathlib import Path
import random
import argparse
from tqdm import tqdm

def collect_upti2_samples(image_base, gt_base, split='train', max_samples=20000):
    """
    Collect UPTI2 samples and create mappings.

    Structure:
    - Images: image_base/split/1/[Font Name]/[degradation]/1.png
    - GT: gt_base/split/1.txt

    Each txt file corresponds to one PNG with that number.
    """
    samples = []

    split_img_path = Path(image_base) / split
    split_gt_path = Path(gt_base) / split

    if not split_img_path.exists():
        print(f"Error: Image path does not exist: {split_img_path}")
        return samples

    if not split_gt_path.exists():
        print(f"Error: Ground truth path does not exist: {split_gt_path}")
        return samples

    # Get all ground truth files
    gt_files = sorted(split_gt_path.glob("*.txt"))
    print(f"Found {len(gt_files)} ground truth files in {split}")

    for gt_file in tqdm(gt_files, desc=f"Processing {split}"):
        # Get the number from filename (e.g., "1.txt" -> "1")
        img_number = gt_file.stem

        # Read transcription
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                transcription = f.read().strip()
        except Exception as e:
            print(f"Error reading {gt_file}: {e}")
            continue

        if not transcription:
            continue

        # Find corresponding images in all fonts and degradation levels
        img_dir = split_img_path / img_number

        if not img_dir.exists():
            continue

        # Traverse through all fonts and degradation levels
        for font_dir in img_dir.iterdir():
            if not font_dir.is_dir():
                continue

            font_name = font_dir.name

            for deg_dir in font_dir.iterdir():
                if not deg_dir.is_dir():
                    continue

                degradation = deg_dir.name

                # ONLY use nodegradation images (skip high/low/medium)
                if degradation != 'nodegradation':
                    continue

                # Look for the image file
                img_file = deg_dir / f"{img_number}.png"

                if img_file.exists():
                    # Create a unique writer ID based on font and degradation
                    # Format: font_deg_imgnum (e.g., AlviNastaleeq_nodeg_1)
                    font_clean = font_name.replace(' ', '')
                    deg_clean = degradation.replace(' ', '')
                    writer_id = f"{font_clean}_{deg_clean}_{img_number}"

                    # Create image name
                    image_name = f"{writer_id}.png"

                    samples.append({
                        'writer_id': writer_id,
                        'image_name': image_name,
                        'transcription': transcription,
                        'original_path': str(img_file),
                        'font': font_name,
                        'degradation': degradation,
                        'img_number': img_number
                    })

    # Shuffle and limit samples
    random.shuffle(samples)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
        print(f"Limited to {max_samples} samples")

    return samples


def organize_dataset(samples, output_base, split='train'):
    """
    Organize images into the expected structure and create annotation file.

    Expected structure:
    - output_base/split/image_name.png  (all images flat)
    - output_base/train.txt or val.txt
    """
    split_dir = Path(output_base) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = Path(output_base) / f"{split}.txt"

    print(f"\nCopying images to {split_dir}...")

    with open(annotation_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples, desc=f"Organizing {split}"):
            # Copy image
            src_path = sample['original_path']
            dst_path = split_dir / sample['image_name']

            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
                continue

            # Write annotation: writer_id,image_name transcription
            line = f"{sample['writer_id']},{sample['image_name']} {sample['transcription']}\n"
            f.write(line)

    print(f"Created annotation file: {annotation_file}")
    print(f"Total images copied: {len(samples)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare UPTI2 dataset for oneDM")
    parser.add_argument('--image_base', type=str, required=True,
                        help='Base path to UPTI2 images (e.g., /storage/1/saima/images_upti2_2/images)')
    parser.add_argument('--gt_base', type=str, required=True,
                        help='Base path to ground truth files')
    parser.add_argument('--output_base', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--max_train', type=int, default=20000,
                        help='Maximum training samples')
    parser.add_argument('--max_val', type=int, default=2000,
                        help='Maximum validation samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    random.seed(args.seed)

    print("="*60)
    print("UPTI2 Dataset Preparation for oneDM")
    print("="*60)

    # Process training set
    print("\n[1/2] Processing Training Set...")
    train_samples = collect_upti2_samples(
        args.image_base,
        args.gt_base,
        split='train',
        max_samples=args.max_train
    )

    if train_samples:
        organize_dataset(train_samples, args.output_base, split='train')
    else:
        print("No training samples found!")

    # Process validation set (if exists)
    print("\n[2/2] Processing Validation Set...")
    val_samples = collect_upti2_samples(
        args.image_base,
        args.gt_base,
        split='test',  # UPTI2 uses 'test' instead of 'val'
        max_samples=args.max_val
    )

    if val_samples:
        organize_dataset(val_samples, args.output_base, split='val')
    else:
        print("No validation samples found!")

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Output directory: {args.output_base}")
    print("="*60)


if __name__ == '__main__':
    main()
