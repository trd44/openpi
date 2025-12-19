import os
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp

# ==========================================
# CONFIGURATION
# ==========================================
# Make sure these point to the 'params' folder inside your checkpoints
LORA_PATH = "/home/train/vla/CyclicLxM/openpi/checkpoints/pi0_cube_sorting_lora/pi0_cube_sorting_lora/29999/params"
MERGED_PATH = "/home/train/vla/CyclicLxM/openpi/checkpoints/pi0_cube_sorting_merged/params"

def compare_weights():
    mngr = ocp.StandardCheckpointer()
    
    print(f"--- Loading LoRA from: {LORA_PATH} ---")
    lora_root = mngr.restore(LORA_PATH)
    
    print(f"--- Loading Merged from: {MERGED_PATH} ---")
    merged_root = mngr.restore(MERGED_PATH)

    print("\n--- RESULTS ---")

    def get_layer(root, expert_path, layer_name):
        # Navigates the PyTree to find the specific layer
        # Adjusts automatically if 'params' is the top key or not
        node = root
        if 'params' in node: node = node['params']
        return node['PaliGemma']['llm']['layers'][expert_path][layer_name]

    def check_layer(expert_name, layer_name):
        try:
            # 1. Get Original Components
            l_node = get_layer(lora_root, expert_name, layer_name)
            w_base = l_node['w']['value']
            a = l_node['lora_a']['value']
            b = l_node['lora_b']['value']

            # 2. Get Merged Weight
            m_node = get_layer(merged_root, expert_name, layer_name)
            w_merged = m_node['w']['value']

            # 3. Calculate Expected Value (Base + A@B)
            # We cast to float32 to ensure the comparison is fair
            delta = jnp.matmul(a.astype(jnp.float32), b.astype(jnp.float32))
            expected = w_base.astype(jnp.float32) + delta

            # 4. Compare
            diff = jnp.mean(jnp.abs(expected - w_merged.astype(jnp.float32)))
            
            print(f"[{expert_name}] {layer_name}")
            print(f"  Mean Difference: {diff:.8f}")
            
            if diff < 0.001:
                print("  ✅ PASS: Weights are identical.")
            else:
                print("  ❌ FAIL: Weights mismatch (The merge math or scaling is wrong).")

        except KeyError as e:
            print(f"  ⚠️ SKIPPED: Could not find key {e}")

    # Test Expert 0 (2B Vision Backbone)
    check_layer('attn', 'q_einsum')

    # Test Expert 1 (300M Action Expert)
    # If this fails or is skipped, the merge script missed the action expert
    check_layer('attn_1', 'q_einsum_1')

if __name__ == "__main__":
    compare_weights()