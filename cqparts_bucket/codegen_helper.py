#!/usr/bin/env python3
"""
VLM Code Generation Helper

Simple Python interface for calling the /codegen endpoint.

Usage:
    python codegen_helper.py reference.jpg --prompt "Make it wider"
    python codegen_helper.py ref.jpg --snapshot current.png --prompt "Add more wheels"
"""

import argparse
import requests
import sys
import os
from pathlib import Path


def call_codegen(
    reference_path: str,
    snapshot_path: str = None,
    prompt: str = "",
    server_url: str = "http://localhost:5160"
) -> dict:
    """
    Call the VLM codegen endpoint.
    
    Args:
        reference_path: Path to reference image (target design)
        snapshot_path: Path to current CAD snapshot (optional)
        prompt: User intent/feedback text
        server_url: Server URL (default: http://localhost:5160)
    
    Returns:
        dict: Response from server
    """
    
    # Validate files
    ref_path = Path(reference_path)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    
    files = {
        'reference': open(ref_path, 'rb')
    }
    
    if snapshot_path:
        snap_path = Path(snapshot_path)
        if not snap_path.exists():
            print(f"Warning: Snapshot image not found: {snapshot_path}", file=sys.stderr)
        else:
            files['snapshot'] = open(snap_path, 'rb')
    
    data = {}
    if prompt:
        data['prompt'] = prompt
    
    print(f"📤 Calling {server_url}/codegen...")
    print(f"   Reference: {reference_path}")
    if snapshot_path:
        print(f"   Snapshot: {snapshot_path}")
    if prompt:
        print(f"   Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    
    try:
        response = requests.post(
            f"{server_url}/codegen",
            files=files,
            data=data,
            timeout=300  # 5 minutes for VLM processing
        )
        
        # Close files
        for f in files.values():
            f.close()
        
        return response.json()
    
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": "Request timed out (VLM took too long to respond)"
        }
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "error": f"Could not connect to server at {server_url}. Is it running?"
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"Unexpected error: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate modified robot_base.py using VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with reference image only
  python codegen_helper.py reference.jpg
  
  # With user prompt
  python codegen_helper.py ref.jpg --prompt "Make the base 50mm longer"
  
  # With snapshot for comparison
  python codegen_helper.py ref.jpg --snapshot current.png --prompt "Match the reference proportions"
  
  # With detailed intent
  python codegen_helper.py target.jpg --prompt "
    The reference shows a more compact design with:
    - Length reduced from 280mm to 220mm
    - 3 wheels per side instead of 2
    - Smaller wheel diameter (~80mm)
  "
  
  # Custom server URL
  python codegen_helper.py ref.jpg --server http://192.168.1.100:5160
        """
    )
    
    parser.add_argument(
        'reference',
        help='Path to reference image (target design)'
    )
    
    parser.add_argument(
        '--snapshot', '-s',
        help='Path to current CAD snapshot image (optional)'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        default="",
        help='User intent/feedback text describing desired changes'
    )
    
    parser.add_argument(
        '--server',
        default='http://localhost:5160',
        help='Server URL (default: http://localhost:5160)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Copy generated code to this path (in addition to generated/)'
    )
    
    args = parser.parse_args()
    
    # Call the API
    result = call_codegen(
        reference_path=args.reference,
        snapshot_path=args.snapshot,
        prompt=args.prompt,
        server_url=args.server
    )
    
    # Display results
    print("\n" + "="*80)
    if result.get('ok'):
        print("✅ SUCCESS!")
        print(f"\n📝 Generated code saved to:")
        print(f"   Main: {result.get('module_path')}")
        print(f"   Backup: {result.get('backup_path')}")
        print(f"\n📊 Stats:")
        print(f"   Code length: {result.get('code_length')} characters")
        print(f"   GLB updated: {result.get('glb_updated')}")
        
        if result.get('message'):
            print(f"\n💡 {result.get('message')}")
        
        # Optionally copy to custom output path
        if args.output:
            try:
                module_path = result.get('module_path')
                if module_path and os.path.exists(module_path):
                    with open(module_path, 'r') as src:
                        code = src.read()
                    with open(args.output, 'w') as dst:
                        dst.write(code)
                    print(f"\n📋 Also copied to: {args.output}")
            except Exception as e:
                print(f"\n⚠️  Could not copy to {args.output}: {e}")
        
        print("\n" + "="*80)
        return 0
    else:
        print("❌ FAILED!")
        print(f"\n🔴 Error: {result.get('error')}")
        
        if 'reject_path' in result:
            print(f"\n📄 Raw VLM output saved to: {result.get('reject_path')}")
            print(f"   Raw length: {result.get('raw_length')} chars")
        
        if 'trace' in result:
            print(f"\n🐛 Stack trace:")
            print(result.get('trace'))
        
        print("\n" + "="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())

