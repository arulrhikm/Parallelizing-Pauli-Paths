#!/usr/bin/env python3
"""
Update test word counts to use the new higher limit (6400 max usable words)
"""

import re

# Read the file
with open('src/tests.cpp', 'r') as f:
    content = f.read()

# Update test word counts to be larger (but under 6400 limit)
replacements = [
    # Test 23: 500 -> 5000
    (r'int num_words = 500; // 500 words - fits within GPU limit of 640',
     'int num_words = 5000; // 5K words - good parallelism'),
    (r'"MultiBlock Test A: 28q, 500 words, 3K l',
     '"MultiBlock Test A: 28q, 5K words, 3K l'),
    
    # Test 24: 300 -> 3000
    (r'int num_words = 300; // 300 words with deep circuit',
     'int num_words = 3000; // 3K words with deep circuit'),
    (r'"MultiBlock Test B: 30q, 300 words, 100K',
     '"MultiBlock Test B: 30q, 3K words, 100K'),
    
    # Test 25: 400 -> 4000
    (r'int num_words = 400; // 400 words for expansion test',
     'int num_words = 4000; // 4K words for expansion test'),
    (r'"MultiBlock Test C: 30q, 400 words, 4K l',
     '"MultiBlock Test C: 30q, 4K words, 4K l'),
    
    # Test 26: 550 -> 5500
    (r'int num_words = 550; // 550 words for balanced test',
     'int num_words = 5500; // 5.5K words for balanced test'),
    (r'"MultiBlock Test D: 30q, 550 words, 5K l',
     '"MultiBlock Test D: 30q, 5.5K words, 5K l'),
    
    # Test 27: 600 -> 6000
    (r'int num_words = 600; // 600 words - near GPU limit',
     'int num_words = 6000; // 6K words - high parallelism'),
    (r'"MultiBlock Test E: 32q, 600 words, 6K l',
     '"MultiBlock Test E: 32q, 6K words, 6K l'),
    
    # Test 28: 450 -> 4500
    (r'int num_words = 450; // 450 words',
     'int num_words = 4500; // 4.5K words'),
    (r'"MultiBlock Test F: 28q, 450 words, 3.5K',
     '"MultiBlock Test F: 28q, 4.5K words, 3.5K'),
    
    # Test 29: 350 -> 3500
    (r'int num_words = 350; // 350 words',
     'int num_words = 3500; // 3.5K words'),
    (r'"MultiBlock Test G: 30q, 350 words, 8K',
     '"MultiBlock Test G: 30q, 3.5K words, 8K'),
    
    # Test 30: 250 -> 2500
    (r'int num_words = 250; // 250 words',
     'int num_words = 2500; // 2.5K words'),
    (r'"MultiBlock Test H: 32q, 250 words, 150K',
     '"MultiBlock Test H: 32q, 2.5K words, 150K'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open('src/tests.cpp', 'w') as f:
    f.write(content)

print("✓ Updated all test word counts (10x increase):")
print("  Test 23: 500 -> 5,000 words")
print("  Test 24: 300 -> 3,000 words")
print("  Test 25: 400 -> 4,000 words")
print("  Test 26: 550 -> 5,500 words")
print("  Test 27: 600 -> 6,000 words")
print("  Test 28: 450 -> 4,500 words")
print("  Test 29: 350 -> 3,500 words")
print("  Test 30: 250 -> 2,500 words")
print("\nNew GPU limit: 6,400 words (was 640)")
