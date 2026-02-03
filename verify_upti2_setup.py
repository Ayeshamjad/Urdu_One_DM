#!/usr/bin/env python3
"""
Quick verification script to check if UPTI2 dataset is properly prepared.
Run this after data preparation but before training.
"""

import os
import cv2
from pathlib import Path
import sys

def check_file_exists(path, description):
    """Check if a file or directory exists."""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {path}")
        return False

def check_annotation_file(file_path, split_name):
    """Check annotation file format and contents."""
    if not Path(file_path).exists():
        print(f"✗ {split_name}.txt not found: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) == 0:
            print(f"✗ {split_name}.txt is empty!")
            return False

        print(f"✓ {split_name}.txt: {len(lines)} samples")

        # Check format of first few lines
        print(f"  Sample lines from {split_name}.txt:")
        for i, line in enumerate(lines[:3]):
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                writer_img, text = parts
                print(f"    {i+1}. {writer_img} -> {text[:30]}...")
            else:
                print(f"    {i+1}. [FORMAT ERROR] {line[:50]}...")

        return True
    except Exception as e:
        print(f"✗ Error reading {split_name}.txt: {e}")
        return False

def check_images(img_dir, annotation_file, split_name, max_check=10):
    """Check if images referenced in annotation file exist."""
    if not Path(img_dir).exists():
        print(f"✗ Image directory not found: {img_dir}")
        return False

    if not Path(annotation_file).exists():
        return False

    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        missing = 0
        checked = 0

        for line in lines[:max_check]:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue

            writer_img = parts[0]
            img_name = writer_img.split(',')[1] if ',' in writer_img else writer_img
            img_path = Path(img_dir) / img_name

            checked += 1
            if not img_path.exists():
                missing += 1
                print(f"  ✗ Missing: {img_name}")

        if missing == 0:
            print(f"✓ Sample images check passed ({checked}/{checked} found)")
            return True
        else:
            print(f"✗ {missing}/{checked} sample images are missing!")
            return False
    except Exception as e:
        print(f"✗ Error checking images: {e}")
        return False

def check_image_dimensions(img_dir, split_name, max_check=10):
    """Check if images have correct dimensions (height=64)."""
    if not Path(img_dir).exists():
        return False

    image_files = list(Path(img_dir).glob("*.png"))[:max_check]

    if len(image_files) == 0:
        print(f"✗ No images found in {img_dir}")
        return False

    correct = 0
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None and img.shape[0] == 64:
            correct += 1

    if correct == len(image_files):
        print(f"✓ Image dimensions check passed ({correct}/{len(image_files)} images have height=64)")
        return True
    else:
        print(f"✗ Some images don't have correct dimensions ({correct}/{len(image_files)} correct)")
        return False

def main():
    print("="*60)
    print("UPTI2 Dataset Setup Verification")
    print("="*60)

    base_dir = "/storage/1/saima/oneDM/Arabic-One-DM"

    all_checks = []

    # Check 1: Annotation files
    print("\n[1] Checking annotation files...")
    all_checks.append(check_annotation_file(f"{base_dir}/data/train.txt", "train"))
    all_checks.append(check_annotation_file(f"{base_dir}/data/val.txt", "val"))

    # Check 2: Processed images
    print("\n[2] Checking processed images...")
    all_checks.append(check_file_exists(
        "/storage/1/saima/oneDM/upti2_processed/train",
        "Processed train images"
    ))
    all_checks.append(check_file_exists(
        "/storage/1/saima/oneDM/upti2_processed/val",
        "Processed val images"
    ))

    # Check 3: Laplace images
    print("\n[3] Checking Laplace edge images...")
    all_checks.append(check_file_exists(
        "/storage/1/saima/oneDM/upti2_processed_laplace/train",
        "Laplace train images"
    ))
    all_checks.append(check_file_exists(
        "/storage/1/saima/oneDM/upti2_processed_laplace/val",
        "Laplace val images"
    ))

    # Check 4: Image references match
    print("\n[4] Checking if annotation references match actual images...")
    all_checks.append(check_images(
        "/storage/1/saima/oneDM/upti2_processed/train",
        f"{base_dir}/data/train.txt",
        "train",
        max_check=10
    ))

    # Check 5: Image dimensions
    print("\n[5] Checking image dimensions...")
    all_checks.append(check_image_dimensions(
        "/storage/1/saima/oneDM/upti2_processed/train",
        "train",
        max_check=10
    ))

    # Check 6: Config file
    print("\n[6] Checking configuration file...")
    all_checks.append(check_file_exists(
        f"{base_dir}/configs/UPTI2_urdu.yml",
        "Config file"
    ))

    # Check 7: unifont pickle
    print("\n[7] Checking unifont glyph data...")
    all_checks.append(check_file_exists(
        f"{base_dir}/data/unifont.pickle",
        "Unifont pickle file"
    ))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(all_checks)
    total = len(all_checks)

    print(f"Checks passed: {passed}/{total}")

    if passed == total:
        print("\n✓✓✓ All checks passed! Ready to start training. ✓✓✓")
        print("\nTo start training, run:")
        print("  bash run_upti2_training.sh")
        print("Or manually:")
        print("  cd /storage/1/saima/oneDM/Arabic-One-DM")
        print("  python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 \\")
        print("    train.py --cfg configs/UPTI2_urdu.yml \\")
        print("    --stable_dif_path runwayml/stable-diffusion-v1-5 --log upti2_training")
        sys.exit(0)
    else:
        print(f"\n✗✗✗ {total - passed} checks failed. Please fix the issues above. ✗✗✗")
        sys.exit(1)

if __name__ == '__main__':
    main()
