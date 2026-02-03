#!/usr/bin/env python3
"""
Check if the current Arabic character set in loader_ara.py covers Urdu characters.
Urdu uses additional characters beyond standard Arabic.
"""

# Current Arabic chars from loader_ara.py
current_arabic_chars = "ءأإآابتثجحخدذرزسشصضطظعغفقكلمنهويىئؤة"

# Additional Urdu-specific characters
urdu_specific_chars = "ٹڈڑںےپچژکگیہ"

# Combined for Urdu
urdu_chars = current_arabic_chars + urdu_specific_chars

print("="*60)
print("Urdu Character Coverage Check")
print("="*60)

print("\nCurrent Arabic characters in loader_ara.py:")
print(f"  {current_arabic_chars}")
print(f"  Total: {len(current_arabic_chars)} characters")

print("\nUrdu-specific additional characters needed:")
print(f"  {urdu_specific_chars}")
print(f"  Total: {len(urdu_specific_chars)} characters")

print("\nComplete Urdu character set:")
print(f"  {urdu_chars}")
print(f"  Total: {len(urdu_chars)} characters")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
print("The current loader_ara.py uses standard Arabic characters.")
print("For UPTI2 Urdu dataset, you may need to add Urdu-specific characters:")
print(f"  ٹ ڈ ڑ ں ے پ چ ژ ک گ ی ہ")
print("\nIf your dataset contains these characters, you should modify")
print("loader_ara.py line 39 to include them:")
print()
print("  arabic_chars = \"ءأإآابتثجحخدذرزسشصضطظعغفقكلمنهويىئؤةٹڈڑںےپچژکگیہ\"")
print()
print("="*60)
