#!/usr/bin/env python3
"""
Analyze text lengths in train.txt to verify max_len setting
"""
import unicodedata
import sys

def strip_harakat(text):
    """Remove Arabic diacritics"""
    harakat_set = set("ًٌٍَُِّْ")
    decomposed = unicodedata.normalize("NFD", text)
    out_chars = []
    for ch in decomposed:
        if unicodedata.combining(ch) and ch in harakat_set:
            continue
        out_chars.append(ch)
    return unicodedata.normalize("NFC", "".join(out_chars))

def effective_length(text):
    """Count characters excluding diacritics"""
    decomposed = unicodedata.normalize("NFD", text)
    return len([ch for ch in decomposed if not unicodedata.combining(ch)])

# Analyze train.txt
train_file = "data/train.txt"

lengths = []
try:
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
            transcription = parts[1]
            transcription = strip_harakat(transcription)
            elen = effective_length(transcription)
            lengths.append(elen)
except FileNotFoundError:
    print(f"Error: Could not find {train_file}")
    print("Run this script from /storage/1/saima/oneDM/Arabic-One-DM/")
    sys.exit(1)

print("=" * 60)
print("Text Length Analysis")
print("=" * 60)
print(f"Total samples in train.txt: {len(lengths)}")
print(f"Min length: {min(lengths)}")
print(f"Max length: {max(lengths)}")
print(f"Mean length: {sum(lengths) / len(lengths):.2f}")
print(f"Median length: {sorted(lengths)[len(lengths)//2]}")
print()

# Count samples by length threshold
thresholds = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
print("Samples that would be loaded with different max_len values:")
print("-" * 60)
for thresh in thresholds:
    count = sum(1 for l in lengths if l <= thresh)
    percentage = (count / len(lengths)) * 100
    print(f"max_len={thresh:2d}: {count:5d} samples ({percentage:5.1f}%)")
print()

# Recommend max_len
for thresh in [40, 50, 60, 70, 80]:
    count = sum(1 for l in lengths if l <= thresh)
    percentage = (count / len(lengths)) * 100
    if percentage >= 95.0:
        print(f"✓ RECOMMENDED: max_len={thresh} captures {percentage:.1f}% of data")
        break

print("\nLength distribution (first 100 chars):")
print("-" * 60)
buckets = {}
for l in lengths:
    bucket = (l // 5) * 5  # 5-char buckets
    buckets[bucket] = buckets.get(bucket, 0) + 1

for bucket in sorted(buckets.keys())[:20]:  # Show first 20 buckets (0-100)
    bar = "#" * (buckets[bucket] // 100 + 1)
    print(f"{bucket:3d}-{bucket+4:3d}: {buckets[bucket]:5d} {bar}")
