#!/usr/bin/env python3
"""
Automatically patch all .wrapped calls for CadQuery 2.x compatibility
"""

import os
import re
import sys

def patch_file(filepath):
    """Patch .wrapped calls in a single file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: simple .wrapped access
    # other.wrapped → (other.wrapped if hasattr(other, 'wrapped') else other)
    content = re.sub(
        r'(\w+)\.wrapped(?!\s*=)',  # Don't match assignments
        r'(\\1.wrapped if hasattr(\\1, "wrapped") else \\1)',
        content
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    base_dir = '/home/ec2-user/Documents/cad-optimizer/cqparts_bucket'
    
    # Directories to patch
    dirs_to_patch = [
        os.path.join(base_dir, 'cqparts'),
        os.path.join(base_dir, 'cqparts_fasteners'),
        os.path.join(base_dir, 'cqparts_bearings'),
    ]
    
    patched = 0
    for dir_path in dirs_to_patch:
        for root, dirs, files in os.walk(dir_path):
            # Skip test directories
            if 'test' in root or '__pycache__' in root:
                continue
                
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    if patch_file(filepath):
                        print(f"✓ Patched: {filepath}")
                        patched += 1
    
    print(f"\n✅ Patched {patched} files")
    return 0

if __name__ == '__main__':
    sys.exit(main())

