#!/usr/bin/env python3
"""
Preprocess UPTI2 images for oneDM training:
1. Resize images to height=64 with aspect ratio preserved
2. Create style reference images (same as training images)
3. Create Laplace edge images for style conditioning
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool

TARGET_H = 512
MIN_W = 512
MAX_W = 1024
ROUND_TO = 256


def resize_keep_h(img):
    """Resize image so that height=TARGET_H and width rounded to multiple of ROUND_TO."""
    h, w = img.shape[:2]
    new_w = int(round(w * TARGET_H / h))
    new_w = max(MIN_W, min(MAX_W, new_w))
    # Round up to next multiple of ROUND_TO
    new_w = (new_w + ROUND_TO - 1) // ROUND_TO * ROUND_TO
    return cv2.resize(img, (new_w, TARGET_H), cv2.INTER_AREA)


def apply_laplace(img):
    """Apply Laplace edge detection."""
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply Laplace operator
    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = np.uint8(np.absolute(laplace))

    return laplace


def process_one_image(args):
    """Process a single image: resize and create laplace version."""
    src_path, dst_img_path, dst_laplace_path = args

    try:
        # Read image
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read {src_path}")
            return False

        # Resize
        resized = resize_keep_h(img)

        # Save resized image
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_img_path), resized)

        # Create and save Laplace version
        laplace = apply_laplace(resized)
        dst_laplace_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_laplace_path), laplace)

        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def preprocess_split(data_dir, split, output_dir, n_threads=8):
    """Preprocess all images in a split."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}")
        return

    # Output directories
    out_img_dir = Path(output_dir) / split
    out_laplace_dir = Path(output_dir + "_laplace") / split

    # Collect all image files
    image_files = list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images in {split}")

    # Prepare jobs
    jobs = []
    for img_file in image_files:
        dst_img = out_img_dir / img_file.name
        dst_laplace = out_laplace_dir / img_file.name
        jobs.append((img_file, dst_img, dst_laplace))

    # Process in parallel
    print(f"Processing {split} images with {n_threads} threads...")
    with Pool(n_threads) as pool:
        results = list(tqdm(
            pool.imap(process_one_image, jobs),
            total=len(jobs),
            desc=f"Processing {split}"
        ))

    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(jobs)} images in {split}")


def create_style_references(data_dir, splits=['train', 'val']):
    """
    Create style reference copies.
    In oneDM, style images are sampled from the same pool as training images.
    So we just copy the processed images to style directories.
    """
    for split in splits:
        split_dir = Path(data_dir) / split
        style_dir = Path(data_dir) / split

        if split_dir.exists():
            # Style images are the same as training images in this case
            print(f"\nStyle references for {split}: using same directory")
        else:
            print(f"Split directory not found: {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess UPTI2 images for oneDM")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train/ and val/ subdirectories with images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed images')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of parallel threads')

    args = parser.parse_args()

    print("="*60)
    print("UPTI2 Image Preprocessing for oneDM")
    print("="*60)

    # Process train and val splits
    for split in ['train', 'val']:
        preprocess_split(args.data_dir, split, args.output_dir, args.threads)

    print("\n" + "="*60)
    print("Image preprocessing complete!")
    print(f"Processed images: {args.output_dir}")
    print(f"Laplace images: {args.output_dir}_laplace")
    print("="*60)


if __name__ == '__main__':
    main()
