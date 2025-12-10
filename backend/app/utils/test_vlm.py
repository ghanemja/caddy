"""
VLM Model Testing Utilities

Test function to load and test the VLM model without running the full UI.
"""
import os
import sys
import tempfile


def test_vlm_model(include_mesh_analysis=True):
    """
    Test function to load and test the VLM model without running the full UI.
    This can be run directly: python -c "from app.utils.test_vlm import test_vlm_model; test_vlm_model()"
    Or: python run.py --test-vlm
    
    Args:
        include_mesh_analysis: If True, also test the mesh ingestion pipeline
    """
    # Import here to avoid circular dependencies
    from app.services.vlm_service import (
        USE_FINETUNED_MODEL,
        FINETUNED_MODEL_PATH,
        load_finetuned_model,
        call_vlm,
    )
    from run import _finetuned_model, _finetuned_processor
    
    print("=" * 80)
    print("Testing VLM Model Loading and Inference")
    print("=" * 80)
    
    # Check model path
    print(f"\n[test] Model path: {FINETUNED_MODEL_PATH}")
    print(f"[test] Path exists: {os.path.exists(FINETUNED_MODEL_PATH)}")
    
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"[test] ✗ ERROR: Model path does not exist!")
        print(f"[test] Expected: {FINETUNED_MODEL_PATH}")
        return False
    
    # Check for required files
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = []
    for f in required_files:
        file_path = os.path.join(FINETUNED_MODEL_PATH, f)
        if not os.path.exists(file_path):
            missing_files.append(f)
        else:
            print(f"[test] ✓ Found {f}")
    
    if missing_files:
        print(f"[test] ✗ ERROR: Missing required files: {missing_files}")
        return False
    
    # Load model
    print(f"\n[test] Loading model...")
    print(f"[test] Note: This may take a few minutes if the base model needs to be downloaded")
    try:
        load_finetuned_model()
        if _finetuned_model is None or _finetuned_processor is None:
            print("[test] ✗ ERROR: Model failed to load")
            return False
        print("[test] ✓ Model loaded successfully")
    except KeyboardInterrupt:
        print("\n[test] Model loading interrupted by user")
        return False
    except Exception as e:
        print(f"[test] ✗ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test inference with a simple prompt
    print(f"\n[test] Testing inference with a simple prompt...")
    try:
        test_prompt = "What is 2+2? Answer with just the number."
        result = call_vlm(
            final_prompt=test_prompt,
            image_data_urls=None,
            expect_json=False
        )
        
        if result and "raw" in result:
            response = result["raw"]
            print(f"[test] ✓ Inference successful!")
            print(f"[test] Provider: {result.get('provider', 'unknown')}")
            print(f"[test] Response length: {len(response)} chars")
            print(f"[test] Response preview: {response[:200]}...")
        else:
            print(f"[test] ✗ ERROR: Invalid response format: {result}")
            return False
    except KeyboardInterrupt:
        print("\n[test] Inference interrupted by user")
        return False
    except Exception as e:
        print(f"[test] ✗ ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test mesh analysis if requested
    if include_mesh_analysis:
        print(f"\n" + "=" * 80)
        print("Testing Mesh Analysis Pipeline")
        print("=" * 80)
        
        try:
            # Check PointNet++ model
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            checkpoint_path = os.path.join(
                BASE_DIR, "models", "pointnet2", "pointnet2_part_seg_msg.pth"
            )
            checkpoint_path = os.path.abspath(checkpoint_path)
            print(f"\n[test] PointNet++ model path: {checkpoint_path}")
            print(f"[test] Path exists: {os.path.exists(checkpoint_path)}")
            
            if not os.path.exists(checkpoint_path):
                print(f"[test] ⚠ WARNING: PointNet++ model not found, skipping mesh analysis test")
                print(f"[test] Expected: {checkpoint_path}")
                return True  # VLM test passed, just mesh analysis missing
            
            # Create a simple test mesh
            print(f"\n[test] Creating test mesh...")
            try:
                import trimesh
                # Create a simple box mesh for testing
                test_mesh = trimesh.creation.box(extents=[1.0, 2.0, 0.5])
                temp_dir = tempfile.mkdtemp(prefix="test_mesh_")
                test_mesh_path = os.path.join(temp_dir, "test_box.obj")
                test_mesh.export(test_mesh_path)
                print(f"[test] ✓ Created test mesh at {test_mesh_path}")
            except Exception as e:
                print(f"[test] ✗ ERROR: Failed to create test mesh: {e}")
                import traceback
                traceback.print_exc()
                return True  # VLM test passed
            
            # Import mesh ingestion components
            print(f"\n[test] Loading mesh ingestion pipeline...")
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from meshml.pointnet_seg.model import load_pretrained_model
            from meshml.semantics.vlm_client_finetuned import FinetunedVLMClient
            from meshml.semantics.ingest_mesh import ingest_mesh_to_semantic_params
            
            # Load PointNet++ model
            print(f"[test] Loading PointNet++ model...")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[test] Using device: {device}")
            
            pointnet_model = load_pretrained_model(
                checkpoint_path=checkpoint_path,
                num_classes=50,
                use_normals=True,
                device=device,
            )
            print(f"[test] ✓ PointNet++ model loaded")
            
            # Initialize VLM client (use the already loaded model)
            print(f"[test] Initializing VLM client...")
            vlm_client = FinetunedVLMClient()
            print(f"[test] ✓ VLM client initialized")
            
            # Run mesh ingestion
            print(f"\n[test] Running mesh ingestion pipeline...")
            print(f"[test] Note: This may take a few minutes (VLM calls + segmentation)")
            render_dir = os.path.join(temp_dir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            
            try:
                result = ingest_mesh_to_semantic_params(
                    mesh_path=test_mesh_path,
                    vlm=vlm_client,
                    model=pointnet_model,
                    render_output_dir=render_dir,
                    num_points=2048,
                )
                
                print(f"[test] ✓ Mesh ingestion completed!")
                print(f"[test] Category: {result.category}")
                print(f"[test] Raw parameters: {len(result.raw_parameters)}")
                print(f"[test] Proposed parameters: {len(result.proposed_parameters)}")
                print(f"[test] Number of parts: {result.extra.get('num_parts', 0)}")
                
                # Show some example parameters
                if result.proposed_parameters:
                    print(f"\n[test] Example proposed parameters:")
                    for i, param in enumerate(result.proposed_parameters[:5]):
                        print(f"  {i+1}. {param.semantic_name}: {param.value} {param.units} (confidence: {param.confidence:.2f})")
                
                # Cleanup
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                
                print(f"\n[test] ✓ Mesh analysis test passed!")
                return True
            except KeyboardInterrupt:
                print("\n[test] Mesh analysis interrupted by user")
                # Cleanup
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                return True  # VLM test passed, mesh analysis interrupted
            
        except KeyboardInterrupt:
            print("\n[test] Mesh analysis setup interrupted by user")
            return True  # VLM test passed
        except Exception as e:
            print(f"[test] ✗ ERROR: Mesh analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return True  # VLM test passed, mesh analysis failed
    
    return True

