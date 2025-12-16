import sys
import os
import gguf

def inspect_gguf(model_path):
    print(f"Inspecting GGUF file: {model_path}")
    
    # MONKEYPATCH: Handle unknown quantization types (like 39 for MXFP4)
    try:
        # Attempt to add the unknown type to the Enum if it doesn't exist
        # This prevents the GGUFReader from crashing when it encounters type 39
        if not hasattr(gguf.GGMLQuantizationType, 'MXFP4'):
             # We extend the IntEnum mechanism or simply mock it if possible
             # Since IntEnum is hard to extend, we might just need to wrap the reader
             pass
    except Exception as e:
        print(f"Patching warning: {e}")

    try:
        # We wrap the reader initialization to catch the specific ValueError
        reader = gguf.GGUFReader(model_path)
    except ValueError as e:
        if "not a valid GGMLQuantizationType" in str(e):
            print("\nError: The 'gguf' library encountered an unknown quantization type (likely MXFP4, type 39).")
            print("Attempting to patch the library in memory to bypass this...")
            
            try:
                # Force add the missing value to the Enum's internal map if possible
                # Or re-implement a robust reader. 
                # Simplest hack: The library fails during tensor reading. 
                # We can try to read just the header/metadata manually if the Reader fails.
                
                # Let's try to patch the Enum class directly
                from enum import IntEnum
                class PatchedGGMLQuantizationType(IntEnum):
                    MXFP4 = 39 
                    # ... copy others if needed, but we can't easily replace the class ref inside the module.
                
                # Better approach: Just raw read the file using struct to get metadata
                # GGUF format: Magic, Version, TensorCount, KVCount, ...
                print("Falling back to raw binary inspection for metadata...")
                inspect_gguf_raw(model_path)
                return
            except Exception as patch_e:
                print(f"Could not patch or raw read: {patch_e}")
                return
        else:
             print(f"Error reading GGUF file: {e}")
             return
    except Exception as e:
        print(f"Error reading GGUF file: {e}")
        return

    print("\n--- Model Architecture ---")
    print(f"Architecture: {reader.get_field('general.architecture').parts[-1].tobytes().decode('utf-8')}")
    
    # Try to find layer count
    try:
        block_count = reader.get_field(f"{reader.get_field('general.architecture').parts[-1].tobytes().decode('utf-8')}.block_count")
        if block_count:
             print(f"Layers (Block Count): {block_count.parts[-1][0]}")
    except:
        print("Could not determine layer count directly from standard keys.")

    print(f"\n--- Metadata (First 20 items) ---")
    for i, field in enumerate(reader.fields.values()):
        if i >= 20:
            break
        print(f"{field.name}: {field.parts[-1]}")

    print("\n--- Tensor Info (First 10 Tensors) ---")
    for i, tensor in enumerate(reader.tensors):
        if i >= 10:
            break
        print(f"Tensor: {tensor.name} | Shape: {tensor.shape} | Type: {tensor.tensor_type}")

import struct

def inspect_gguf_raw(path):
    # Minimal GGUF parser to extract KV pairs even if tensors fail
    # Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
    with open(path, "rb") as f:
        # Header
        magic = f.read(4)
        if magic != b"GGUF":
            print("Not a GGUF file (bad magic)")
            return
        
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]
        
        print(f"\n[RAW READ MODE]")
        print(f"GGUF Version: {version}")
        print(f"Tensor Count: {tensor_count}")
        print(f"KV Count: {kv_count}")
        print("\n--- Metadata ---")

        for _ in range(kv_count):
            try:
                # Key
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8")
                
                # Value Type
                val_type = struct.unpack("<I", f.read(4))[0]
                
                # Value (Simplified handling for common types)
                # GGUF_TYPE_UINT8   = 0
                # GGUF_TYPE_INT8    = 1
                # ...
                # GGUF_TYPE_STRING  = 8
                # GGUF_TYPE_ARRAY   = 9
                # ...
                # GGUF_TYPE_UINT32  = 4
                # GGUF_TYPE_INT32   = 5
                # GGUF_TYPE_FLOAT32 = 6
                # GGUF_TYPE_BOOL    = 7
                # GGUF_TYPE_UINT64  = 10
                # GGUF_TYPE_INT64   = 11
                # GGUF_TYPE_FLOAT64 = 12
                
                value = "?"
                if val_type == 8: # String
                    s_len = struct.unpack("<Q", f.read(8))[0]
                    value = f.read(s_len).decode("utf-8", errors="replace")
                elif val_type in [4, 5]: # Int32/Uint32
                    value = struct.unpack("<I", f.read(4))[0]
                elif val_type in [10, 11]: # Int64/Uint64
                    value = struct.unpack("<Q", f.read(8))[0]
                elif val_type == 6: # Float32
                    value = struct.unpack("<f", f.read(4))[0]
                elif val_type == 7: # Bool
                    value = struct.unpack("?", f.read(1))[0]
                elif val_type == 9: # Array (Skip content roughly)
                     arr_type = struct.unpack("<I", f.read(4))[0]
                     arr_len = struct.unpack("<Q", f.read(8))[0]
                     value = f"[Array of {arr_len} items, type {arr_type}]"
                     # We have to actually skip the bytes to get to next key
                     # This is hard without full implementation, so we might stop here
                     print(f"{key}: {value} (Stopping raw read due to array complexity)")
                     break
                else:
                    # Generic skip attempt (risky)
                    # Use predefined sizes if possible, else break
                     print(f"{key}: [Unknown Type {val_type}] (Stopping raw read)")
                     break

                print(f"{key}: {value}")
                
            except Exception as e:
                print(f"Error parsing KV pair: {e}")
                break

if __name__ == "__main__":
    # Default path provided by user
    # Folder: C:\Users\BassemMonla\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF
    # File: gpt-oss-20b-MXFP4.gguf
    default_path = r"C:\Users\BassemMonla\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf"

    target_path = ""
    
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = default_path
        print(f"No argument provided. Using default path: {target_path}")

    if os.path.exists(target_path):
        inspect_gguf(target_path)
    else:
        print(f"Error: File not found at {target_path}")
        print("Please provide a valid path as an argument or update the default_path variable.")
