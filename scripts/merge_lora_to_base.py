import os
import shutil
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp

# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_DIR = "/home/train/vla/CyclicLxM/openpi/checkpoints/pi0_cube_sorting_lora/pi0_cube_sorting_lora/29999"
MERGED_ROOT = "/home/train/vla/CyclicLxM/openpi/checkpoints/pi0_cube_sorting_merged"
OUTPUT_PARAMS_DIR = os.path.join(MERGED_ROOT, "params")

# INCREASE THIS if the robot moves correctly but stops short/hovers.
# Try 2.0 first. If it becomes jittery/violent, lower to 1.5.
SCALING_FACTOR = 1.0

def merge_pi0_checkpoint():
    mngr = ocp.StandardCheckpointer()
    print(f"--- Loading params from: {CHECKPOINT_DIR} ---")
    state = mngr.restore(os.path.join(CHECKPOINT_DIR, "params"))

    def merge_logic(path, node):
        # We need to handle both standard and MLP LoRA structures
        is_standard = isinstance(node, dict) and 'lora_a' in node and 'lora_b' in node
        
        # If it's a dictionary, we iterate
        if isinstance(node, dict) and not is_standard:
            new_node = {}
            processed_keys = set()
            for k, v in node.items():
                if k.endswith('_lora_a'):
                    base_k = k.replace('_lora_a', '')
                    b_key = k.replace('_lora_a', '_lora_b')
                    
                    if base_k in node and b_key in node:
                        # MLP MERGE LOGIC
                        w = node[base_k]['value']
                        a = node[k]['value']
                        b = node[b_key]['value']
                        
                        # Apply Scaling Factor here
                        delta = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)) * SCALING_FACTOR
                        
                        new_node[base_k] = {'value': (w.astype(jnp.float32) + delta).astype(jnp.bfloat16)}
                        processed_keys.update([base_k, k, b_key])
                        print(f"Merged MLP (Boost {SCALING_FACTOR}x): {path}/{base_k}")
                    
                elif k not in processed_keys:
                    new_node[k] = merge_logic(f"{path}/{k}", v)
            return new_node

        # ATTENTION MERGE LOGIC
        if is_standard:
            w = node['w']['value']
            a = node['lora_a']['value']
            b = node['lora_b']['value']
            
            # Apply Scaling Factor here
            delta = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32)) * SCALING_FACTOR
            
            new_w = (w.astype(jnp.float32) + delta).astype(jnp.bfloat16)
            print(f"Merged Attn (Boost {SCALING_FACTOR}x): {path}")
            return {'w': {'value': new_w}}

        return node

    print(f"--- Starting High-Precision Merge with {SCALING_FACTOR}x Boost ---")
    merged_params = merge_logic("params", state)

    print(f"--- Saving Merged Weights to: {OUTPUT_PARAMS_DIR} ---")
    mngr.save(OUTPUT_PARAMS_DIR, merged_params, force=True)
    mngr.wait_until_finished() 

    # Copy assets automatically
    src_assets = os.path.join(CHECKPOINT_DIR, "assets")
    dst_assets = os.path.join(MERGED_ROOT, "assets")
    if os.path.exists(src_assets):
        if os.path.exists(dst_assets): shutil.rmtree(dst_assets)
        shutil.copytree(src_assets, dst_assets)
        print("--- Assets Copied ---")

    print(f"\n--- DONE! Use variant 'gemma_2b' to serve. ---")

if __name__ == "__main__":
    merge_pi0_checkpoint()