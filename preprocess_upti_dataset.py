#!/usr/bin/env python3
"""
Preprocess UPTI dataset images in-place to a target height.
This script resizes all images in the UPTI directory structure while preserving aspect ratio.

UPTI structure:
{images_base}/{split}/{id}/{font}/{degradation}/{id}.png
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool

TARGET_H = 1024


def resize_keep_aspect(img, target_h):
    """Resize image to target height, keeping aspect ratio."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)


def process_one_image(img_path):
    """Resize a single image in-place."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, f"Could not read {img_path}"

        # Check if already correct size
        if img.shape[0] == TARGET_H:
            return True, None

        # Resize
        resized = resize_keep_aspect(img, TARGET_H)

        # Save in-place
        cv2.imwrite(str(img_path), resized)
        return True, None
    except Exception as e:
        return False, f"Error processing {img_path}: {e}"


def preprocess_upti_split(images_base, split, n_threads=8):
    """Preprocess all images in a UPTI split."""
    split_dir = Path(images_base) / split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return

    # Collect all PNG images recursively
    image_files = list(split_dir.rglob("*.png"))
    print(f"\nFound {len(image_files)} images in {split}")

    if len(image_files) == 0:
        print(f"No images found in {split_dir}")
        return

    # Process in parallel
    print(f"Resizing images to {TARGET_H}px height with {n_threads} threads...")
    with Pool(n_threads) as pool:
        results = list(tqdm(
            pool.imap(process_one_image, image_files),
            total=len(image_files),
            desc=f"Processing {split}"
        ))

    success_count = sum(1 for success, _ in results if success)
    print(f"Successfully processed {success_count}/{len(image_files)} images in {split}")

    # Show errors if any
    errors = [msg for success, msg in results if not success and msg]
    if errors:
        print(f"\nErrors encountered:")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors)-10} more errors")


def main():
    parser = argparse.ArgumentParser(description="Resize UPTI dataset images to target height")
    parser.add_argument('--images_base', type=str, required=True,
                        help='Base directory containing train/ and val/ subdirectories')
    parser.add_argument('--target_h', type=int, default=1024,
                        help='Target height for images (default: 1024)')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of parallel threads')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to process (default: train val)')

    args = parser.parse_args()

    global TARGET_H
    TARGET_H = args.target_h

    print("="*70)
    print(f"UPTI Dataset Image Resizing to {TARGET_H}px height")
    print("="*70)
    print(f"Images base: {args.images_base}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"NOTE: Images will be resized IN-PLACE (original files will be overwritten)")
    print("="*70)

    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    # Process each split
    for split in args.splits:
        preprocess_upti_split(args.images_base, split, args.threads)

    print("\n" + "="*70)
    print("Preprocessing complete!")
    print("="*70)


if __name__ == '__main__':
    main()
