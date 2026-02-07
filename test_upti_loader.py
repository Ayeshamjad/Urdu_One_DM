#!/usr/bin/env python3
"""
Test script to verify UPTI data loader works correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader.loader_ara import IAMDataset
import torch

def test_upti_loader():
    print("=" * 70)
    print("Testing UPTI Data Loader")
    print("=" * 70)

    # UPTI configuration
    upti_config = {
        'images_base': '/home/osama/Downloads/images_upti2_2',
        'gt_base': '/home/osama/Downloads/groundtruth_upti2/groundtruth',
        'font': 'Pak Nastaleeq',
        'degradation': 'nodegradation'
    }

    print("\nConfiguration:")
    for key, value in upti_config.items():
        print(f"  {key}: {value}")

    # Create dataset (using dummy paths for image_path, style_path, laplace_path - not used in UPTI mode)
    print("\n" + "-" * 70)
    print("Loading TRAIN dataset...")
    print("-" * 70)

    try:
        train_dataset = IAMDataset(
            image_path='./data/dummy',
            style_path='./data/dummy',
            laplace_path='./data/dummy',
            type='train',
            max_len=160,
            dataset_format='upti',
            upti_config=upti_config
        )
        print(f"✓ Train dataset loaded successfully!")
        print(f"  Number of samples: {len(train_dataset)}")
    except Exception as e:
        print(f"✗ Error loading train dataset: {e}")
        return False

    print("\n" + "-" * 70)
    print("Loading TEST dataset...")
    print("-" * 70)

    try:
        test_dataset = IAMDataset(
            image_path='./data/dummy',
            style_path='./data/dummy',
            laplace_path='./data/dummy',
            type='test',
            max_len=160,
            dataset_format='upti',
            upti_config=upti_config
        )
        print(f"✓ Test dataset loaded successfully!")
        print(f"  Number of samples: {len(test_dataset)}")
    except Exception as e:
        print(f"✗ Error loading test dataset: {e}")
        return False

    # Test loading a sample
    print("\n" + "-" * 70)
    print("Testing sample loading...")
    print("-" * 70)

    try:
        sample = train_dataset[0]
        print(f"✓ Sample loaded successfully!")
        print(f"  Image shape: {sample['img'].shape}")
        print(f"  Style shape: {sample['style'].shape}")
        print(f"  Laplace shape: {sample['laplace'].shape}")
        print(f"  Transcription: {sample['transcr']}")
        print(f"  Writer ID (Font): {sample['wid']}")
        print(f"  Image path: {sample['image_name']}")
    except Exception as e:
        print(f"✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test collate function with batch
    print("\n" + "-" * 70)
    print("Testing batch collation...")
    print("-" * 70)

    try:
        # Create a small batch
        batch_samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
        batch = train_dataset.collate_fn_(batch_samples)

        print(f"✓ Batch created successfully!")
        print(f"  Batch images shape: {batch['img'].shape}")
        print(f"  Batch style shape: {batch['style'].shape}")
        print(f"  Batch laplace shape: {batch['laplace'].shape}")
        print(f"  Batch content shape: {batch['content'].shape}")
        print(f"  Number of samples in batch: {len(batch['transcr'])}")
    except Exception as e:
        print(f"✗ Error creating batch: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("✓ All tests passed! UPTI data loader is working correctly.")
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = test_upti_loader()
    sys.exit(0 if success else 1)
