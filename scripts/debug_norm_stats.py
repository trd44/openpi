"""Debug script to check what actions look like when computing norm stats."""

import numpy as np
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.models.model as _model
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        import numpy as np
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

def main(config_name: str):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Create dataloader WITHOUT normalization (same as compute_norm_stats.py)
    if data_config.rlds_data_dir is not None:
        dataset = _data_loader.create_rlds_dataset(
            data_config, config.model.action_horizon, config.batch_size, shuffle=False
        )
        dataset = _data_loader.IterableTransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                RemoveStrings(),  # Remove strings since they are not supported by JAX
            ],
            is_batched=True,
        )
        num_batches = len(dataset) // config.batch_size if hasattr(dataset, '__len__') else 10
        data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    else:
        dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                RemoveStrings(),  # Remove strings since they are not supported by JAX
            ],
        )
        num_batches = min(10, len(dataset) // config.batch_size)
        data_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=config.batch_size,
            num_workers=0,  # Single worker for debugging
            shuffle=False,
            num_batches=num_batches,
        )
    
    print(f"Action horizon: {config.model.action_horizon}")
    print(f"Action dim: {config.model.action_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Action sequence keys: {data_config.action_sequence_keys}")
    print(f"Checking {num_batches} batches...\n")
    
    # First, let's check the raw dataset before transforms (like inspect_lerobot.py does)
    print("=== Checking raw dataset (direct access, like inspect_lerobot.py) ===")
    if data_config.rlds_data_dir is None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            raw_lerobot_dataset = LeRobotDataset(data_config.repo_id)
            print(f"Raw LeRobot dataset length: {len(raw_lerobot_dataset)}")
            raw_frame = raw_lerobot_dataset[0]
            print(f"Raw frame type: {type(raw_frame)}")
            print(f"Raw frame keys: {raw_frame.keys() if isinstance(raw_frame, dict) else 'Not a dict'}")
            if isinstance(raw_frame, dict):
                for key in data_config.action_sequence_keys:
                    if key in raw_frame:
                        raw_actions = raw_frame[key]
                        raw_actions_np = np.asarray(raw_actions)
                        print(f"\nRaw '{key}' from LeRobotDataset:")
                        print(f"  Shape: {raw_actions_np.shape}")
                        print(f"  Dtype: {raw_actions_np.dtype}")
                        print(f"  First 10 values: {raw_actions_np.flat[:10] if raw_actions_np.size > 0 else 'Empty'}")
                        if len(raw_actions_np.shape) == 1 and raw_actions_np.shape[0] >= 7:
                            print(f"  First 7 values (action dim): {raw_actions_np[:7]}")
                            if raw_actions_np.shape[0] >= 6:
                                print(f"  Rotation components [3:6]: {raw_actions_np[3:6]}")
            print()
        except Exception as e:
            print(f"Could not load raw dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Now check what create_torch_dataset does (with delta_timestamps)
    print("=== Checking dataset after create_torch_dataset (with action_horizon) ===")
    if data_config.rlds_data_dir is None:
        dataset_with_horizon = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        print(f"Dataset with horizon length: {len(dataset_with_horizon)}")
        sample_with_horizon = dataset_with_horizon[0]
        print(f"Sample keys: {sample_with_horizon.keys() if isinstance(sample_with_horizon, dict) else 'Not a dict'}")
        if isinstance(sample_with_horizon, dict):
            for key in data_config.action_sequence_keys:
                if key in sample_with_horizon:
                    actions_horizon = sample_with_horizon[key]
                    actions_horizon_np = np.asarray(actions_horizon)
                    print(f"\n'{key}' after create_torch_dataset (with action_horizon={config.model.action_horizon}):")
                    print(f"  Shape: {actions_horizon_np.shape}")
                    print(f"  Dtype: {actions_horizon_np.dtype}")
                    print(f"  Expected: (action_horizon, action_dim) = ({config.model.action_horizon}, {config.model.action_dim})")
                    if len(actions_horizon_np.shape) >= 2:
                        print(f"  First action in sequence: {actions_horizon_np[0]}")
                        if actions_horizon_np.shape[0] > 0 and actions_horizon_np.shape[1] >= 7:
                            print(f"  First action rotation [3:6]: {actions_horizon_np[0, 3:6]}")
        print()
    
    all_actions = []
    all_actions_reshaped = []
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"\n=== Batch {batch_idx} ===")
        print(f"Batch type: {type(batch)}")
        
        # Check if batch is a tuple (observation, actions) or dict
        if isinstance(batch, tuple):
            print(f"Batch is a tuple with {len(batch)} elements")
            print(f"  Element 0 type: {type(batch[0])}")
            print(f"  Element 1 type: {type(batch[1])}")
            if len(batch) >= 2:
                actions = batch[1]
                print(f"  Using batch[1] as actions")
            else:
                print("  Batch tuple too short!")
                continue
        elif isinstance(batch, dict):
            print(f"Batch is a dict with keys: {list(batch.keys())}")
            actions = batch.get("actions", None)
            if actions is None:
                print("No 'actions' key found!")
                print(f"Available keys: {list(batch.keys())}")
                continue
        else:
            print(f"Batch is not a dict or tuple, it's: {type(batch)}")
            print(f"Batch dir: {[x for x in dir(batch) if not x.startswith('_')]}")
            continue
        
        print(f"Actions type: {type(actions)}")
        print(f"Actions shape: {actions.shape if hasattr(actions, 'shape') else 'No shape'}")
        
        # Convert to numpy
        actions_np = np.asarray(actions)
        print(f"Actions numpy shape: {actions_np.shape}")
        print(f"Actions numpy dtype: {actions_np.dtype}")
        
        # Check what [0] indexing does (as in compute_norm_stats.py line 105)
        # This is the KEY LINE - batch[key][0] - what does [0] do?
        print(f"\n--- Analyzing batch['actions'][0] as in compute_norm_stats.py ---")
        if isinstance(actions, (list, tuple)):
            print(f"actions is list/tuple with length {len(actions)}")
            print(f"First element type: {type(actions[0])}")
            print(f"First element shape: {actions[0].shape if hasattr(actions[0], 'shape') else 'No shape'}")
            actions_0 = actions[0]
        else:
            print(f"actions is not list/tuple, it's: {type(actions)}")
            # Try indexing anyway to see what happens
            try:
                actions_0 = actions[0]
                print(f"actions[0] works! Type: {type(actions_0)}, Shape: {actions_0.shape if hasattr(actions_0, 'shape') else 'No shape'}")
            except (TypeError, IndexError) as e:
                print(f"actions[0] failed: {e}")
            actions_0 = actions
        
        values = np.asarray(actions_0)
        print(f"values (after np.asarray(actions_0)) shape: {values.shape}")
        print(f"values dtype: {values.dtype}")
        
        # Check if this matches what compute_norm_stats expects
        # Expected: (batch_size, action_horizon, action_dim) -> reshape to (-1, action_dim)
        expected_shape = (config.batch_size, config.model.action_horizon, config.model.action_dim)
        print(f"Expected shape (batch, horizon, dim): {expected_shape}")
        if values.shape == expected_shape:
            print("✓ Shape matches expected!")
        else:
            print(f"⚠️  Shape mismatch! Got {values.shape}, expected {expected_shape}")
        
        # Reshape as in compute_norm_stats.py line 106
        values_reshaped = values.reshape(-1, values.shape[-1])
        print(f"values reshaped shape: {values_reshaped.shape}")
        
        # Print some sample values
        print(f"\nFirst 5 action vectors (reshaped):")
        for i in range(min(5, len(values_reshaped))):
            print(f"  {i}: {values_reshaped[i]}")
        
        # Check rotation components
        if values_reshaped.shape[1] >= 6:
            rotation_components = values_reshaped[:, 3:6]
            non_zero_rot = np.any(np.abs(rotation_components) > 1e-6, axis=1)
            if np.any(non_zero_rot):
                print(f"\n⚠️  Found {np.sum(non_zero_rot)} action vectors with non-zero rotation components!")
                print(f"   First non-zero rotation: {rotation_components[non_zero_rot][0]}")
            else:
                print(f"\n✓ All rotation components are zero in this batch")
        
        all_actions.append(values_reshaped)
        
        if batch_idx >= 2:  # Only check first few batches
            break
    
    if all_actions:
        all_actions_concat = np.concatenate(all_actions, axis=0)
        print(f"\n\n=== Summary ===")
        print(f"Total action vectors checked: {len(all_actions_concat)}")
        print(f"Action mean: {np.mean(all_actions_concat, axis=0)}")
        print(f"Action std: {np.std(all_actions_concat, axis=0)}")
        print(f"Action min: {np.min(all_actions_concat, axis=0)}")
        print(f"Action max: {np.max(all_actions_concat, axis=0)}")
        
        if all_actions_concat.shape[1] >= 6:
            rotation_components = all_actions_concat[:, 3:6]
            non_zero_rot = np.any(np.abs(rotation_components) > 1e-6, axis=1)
            print(f"\nRotation components (indices 3:6):")
            print(f"  Mean: {np.mean(rotation_components, axis=0)}")
            print(f"  Std: {np.std(rotation_components, axis=0)}")
            print(f"  Non-zero count: {np.sum(non_zero_rot)} / {len(rotation_components)}")

if __name__ == "__main__":
    import tyro
    tyro.cli(main)

