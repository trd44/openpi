#%%
import os
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polars import first
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from datetime import datetime
#%%

#%% read the directories in /home/train/VLA-probing/policy_records
data_dir = '/home/train/VLA-probing/policy_records'
# list all the directories in data_dir that are directories
task_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
task_list
# %% load a step file to see what the data looks like
step_file = os.path.join(data_dir, task_list[0], 'episode_0_success=True', 'step_251_success=True.npy')
data = np.load(step_file, allow_pickle=True).item()
data['outputs/prefix_out'].shape, data['outputs/prefix_mask'].shape
#%% utility functions
# structure of a step file
'inputs/observation/image', 
'inputs/observation/wrist_image', 
'inputs/observation/state', 
'inputs/observation/subgoal_successes', 
'inputs/observation/goal_success', 
'inputs/timestep', 
'inputs/episode_idx', 
'inputs/prompt', 
'outputs/actions', 
'outputs/prefix_out', 
'outputs/state', 
'outputs/suffix_out_aggregated', 
'outputs/policy_timing/infer_ms'

def load_step_file(step_file, key='outputs/suffix_out_aggregated'):
    # read the numpy file
    data = np.load(step_file, allow_pickle=True).item()
    return data[key]

def load_all_steps(episode_path, key='outputs/suffix_out_aggregated', mask_key=None) -> np.ndarray:
    """
    Load the steps in the episode_path.
    """
    step_files = [f for f in os.listdir(episode_path) if f.endswith('.npy')]
    step_files.sort()  # sort the step files to get the lowest step number first
    steps = []
    for step_file in step_files:
        step_data = load_step_file(os.path.join(episode_path, step_file), key=key)
        # if there is a mask provided, apply the mask to the step_data
        if mask_key is not None:
            mask = load_step_file(os.path.join(episode_path, step_file), key=mask_key).astype(bool)
            step_data = step_data[mask]
        # if the step_data has more than 1 dimension, apply mean pooling on the first dimension
        if isinstance(step_data, np.ndarray) and len(step_data.shape) > 1:
            step_data = step_data.mean(axis=0)
        steps.append(step_data)
    return np.array(steps)

def load_and_sort_on_success(task_path, key='outputs/suffix_out_aggregated', mask_key=None) -> dict[str, np.ndarray]:
    """
    Load all episodes in the task_path and sort them based on success.
    Returns: dict with keys 'success' and 'failure', each containing a list of episodes.
    """

    episode_list = os.listdir(task_path)
    episode_list = [e for e in episode_list if os.path.isdir(os.path.join(task_path, e))]
    episode_list.sort()  # sort the episode list to get the lowest episode number first
    episodes = {'success': [], 'failure': []}
    for episode in episode_list:
        episode_path = os.path.join(task_path, episode)
        if 'success=True' in episode:
            episodes['success'].append(load_all_steps(episode_path, key=key, mask_key=mask_key))
        else:
            episodes['failure'].append(load_all_steps(episode_path, key=key, mask_key=mask_key))
    return episodes

def encode_subgoals(step_subgoals, mode='chronological'):
    """
    Encode the binary subgoal successes array into a single integer.
    """
    if mode == 'chronological':
        # there are 4 subgoals expected to be completed in order
        assert len(step_subgoals) == 4 or len(step_subgoals) == 2, "Expected 4 or 2 subgoals for chronological mode"
        # define a mapping function
        def map_subgoal_state(state):
            if len(state) == 4:
                # if both subgoal 2 and 4 are completed, return 4
                if state[1] == 1 and state[3] == 1:
                    return 4
                # if both subgoal 2 and 3 are completed, return 3
                if state[1] == 1 and state[2] == 1:
                    return 3
                # if only subgoal 2 is completed, return 2
                if state[2] == 1:
                    return 2
                # if only subgoal 1 is completed, return 1
                if state[1] == 1:
                    return 1
                return 0
            elif len(state) == 2:
                if state[1] == 1:
                    return 2
                if state[0] == 1:
                    return 1
                return 0
        step_subgoals = tuple(step_subgoals)
        return map_subgoal_state(step_subgoals)
    
    # reverse the array to have the first subgoal as the least significant bit
    step_subgoals = step_subgoals[::-1]
    return int("".join(str(int(x)) for x in step_subgoals), 2)


def rename_episode_folder():
    """
    Rename the episode folder to append success=True/False
    """
    for task in task_list:
        task_path = os.path.join(data_dir, task)
        if not os.path.isdir(task_path):
            continue
        episode_list = os.listdir(task_path)
        for episode in episode_list:
            episode_path = os.path.join(task_path, episode)
            if not os.path.isdir(episode_path):
                continue
            # check if there's a step_{number}_success=True.npy file in the episode foler
            step_files = os.listdir(episode_path)
            success_files = [f for f in step_files if 'success=True' in f]
            
            # if there is a success file, rename the episode folder to append success=True
            if len(success_files) > 0:
                new_episode_name = f"{episode}_success=True"
            else:
                new_episode_name = f"{episode}_success=False"
            new_episode_path = os.path.join(task_path, new_episode_name)
            os.rename(episode_path, new_episode_path)

def pad_with_nans(episode_list):
    if len(episode_list) == 0:
        return np.array([])
    
    # Find maximum number of steps
    max_steps = max(episode.shape[0] for episode in episode_list)
    
    # Pad all episodes to max_steps with NaNs
    padded_episodes = np.array([np.pad(episode, ((0, max_steps - episode.shape[0]), (0, 0)), mode='constant', constant_values=np.nan) for episode in episode_list])
    return padded_episodes

def unpad_episodes(padded_episodes):
    """
    Remove NaN padding from a list of padded episodes.
    """
    unpadded_episodes = []
    for episode in padded_episodes:
        # Find the index of the last non-NaN row
        if np.isnan(episode).all():
            unpadded_episodes.append(np.array([]))  # If all values are NaN, return an empty array
            continue
        last_valid_index = np.where(~np.isnan(episode).all(axis=1))[0][-1]
        unpadded_episodes.append(episode[:last_valid_index + 1])
    return unpadded_episodes

def save_task_episodes(task_path, episodes):
    """
    Save the episodes dict to a npz file in the task_path.
    """
    save_path = os.path.join(task_path, 'episodes.npz')
    # Pad episodes with NaNs
    episodes['success'] = pad_with_nans(episodes['success'])
    episodes['failure'] = pad_with_nans(episodes['failure'])
    np.savez_compressed(save_path, success=episodes['success'], failure=episodes['failure'])
    print(f"Saved episodes to {save_path}")

def load_and_save_task_episodes_concatenated(task_path, success=True, key='outputs/suffix_out_aggregated', mask_key=None) -> np.ndarray:
    """
    Save the episodes to a npz file in the task_path, concatenated along the time axis.
    """
    save_path = os.path.join(task_path, f'episodes_success={success}_key={key.replace('/', '_')}.npz')
    # first check if the path exists
    if os.path.exists(save_path):
        print(f"Loading existing episodes from {save_path}")
        with np.load(save_path) as data:
            return data['episodes'] 
    else:
        episodes = load_and_sort_on_success(task_path, key=key, mask_key=mask_key)
        # based on the success argument, keep only the success or failure episodes
        if success:
            episodes = episodes['success']
        else:
            episodes = episodes['failure']
        # concatenate all episodes along the time axis
        if len(episodes) == 0:
            concatenated_episodes = np.array([])
        else:
            concatenated_episodes = np.concatenate(episodes, axis=0)
        # if key is 'inputs/observation/subgoal_successes', apply encode_subgoals to each row
        if key == 'inputs/observation/subgoal_successes':
            concatenated_episodes = np.array([encode_subgoals(step) for step in concatenated_episodes])
        np.savez_compressed(save_path, episodes=concatenated_episodes)
        print(f"Saved concatenated episodes to {save_path}")
        return concatenated_episodes
        

def t_SNE(episode_data, perplexity=50, n_components=2, random_state=42):
    """
    Args:
        episode_data: np.ndarray of shape (num_steps, feature_dim)
        perplexity: t-SNE perplexity parameter
        n_components: number of dimensions for t-SNE output
        random_state: random seed for t-SNE
    Returns:
        episode_data_tsne: np.ndarray of shape (num_steps, n_components)
    """
    # Standardize the data
    # scaler = StandardScaler()
    # episode_data_std = scaler.fit_transform(episode_data)
    # reduce dimensionality with PCA to speed up t-SNE
    pca = PCA(n_components=min(50, episode_data.shape[1]))
    episode_data_pca = pca.fit_transform(episode_data)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    episode_data_tsne = tsne.fit_transform(episode_data_pca)
    return episode_data_tsne

def visualize_episode_tsne(episode_data_tsne, episode_labels, title="t-SNE of Episode", save_path=None, cmap="summer", marker='o', cbar_label='Subgoal Progress'):
    """
    Visualize the t-SNE of the episode data.
    Args:
        episode_data_tsne: np.ndarray of shape (num_steps, 2)
        episode_labels: np.array of shape (num_steps,) with labels for each step
        title: title of the plot
        save_path: if provided, save the plot to this path
    """
    #episode_data_tsne = t_SNE(episode_data)

    # set the style as seaborn
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 8))
    # Use plt.scatter instead of sns.scatterplot to create a mappable for colorbar
    scatter = plt.scatter(episode_data_tsne[:, 0], episode_data_tsne[:, 1], c=episode_labels, cmap=cmap, alpha=0.7, marker=marker)

    # Create colorbar with proper ticks
    # if the ticks are too many, set the number of ticks to 10
    if np.max(episode_labels) > 10:
        cbar = plt.colorbar(ticks=np.linspace(np.min(episode_labels), np.max(episode_labels), num=10, dtype=int))
        cbar.ax.set_yticklabels([str(i) for i in np.linspace(np.min(episode_labels), np.max(episode_labels), num=10, dtype=int)])
    else:
        cbar = plt.colorbar(ticks=np.arange(np.min(episode_labels), np.max(episode_labels)+1))
        cbar.ax.set_yticklabels([str(i) for i in range(np.min(episode_labels), np.max(episode_labels)+1)])
    cbar.set_label(cbar_label)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    #plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    plt.show()

def visualize_success_vs_failed_episode_tsne(success_tsne, success_labels, failed_tsne, failed_labels, cmap1='summer', cmap2='autumn', cbar_label1='Subgoal Progress (Success)', cbar_label2='Subgoal Progress (Failure)', title="t-SNE of Success vs Failure", save_path=None):
    """
    Visualize the t-SNE of successful and failed episodes together.
    Args:
        success_tsne: np.ndarray of shape (num_success_steps, 2)
        success_labels: np.array of shape (num_success_steps,) with labels for each step
        failed_tsne: np.ndarray of shape (num_failed_steps, 2)
        failed_labels: np.array of shape (num_failed_steps,) with labels for each step
        title: title of the plot
        save_path: if provided, save the plot to this path
    """
    sns.set(style="darkgrid")

    plt.figure(figsize=(10, 8))
    plt.scatter(success_tsne[:, 0], success_tsne[:, 1], c=success_labels, cmap=cmap1, alpha=0.7, label='Success', marker='o')
    # set a colorbar for the success points
    # if the ticks are too many, set the number of ticks to 10
    if np.max(success_labels) > 10:
        cbar = plt.colorbar(ticks=np.linspace(0, np.max(success_labels), num=10, dtype=int))
        cbar.ax.set_yticklabels([str(i) for i in np.linspace(np.min(success_labels), np.max(success_labels), num=10, dtype=int)])
    else:
        cbar = plt.colorbar(ticks=np.arange(np.min(success_labels), np.max(success_labels)+1))
        cbar.ax.set_yticklabels([str(i) for i in range(np.min(success_labels), np.max(success_labels)+1)])
    cbar.set_label(cbar_label1)
    plt.scatter(failed_tsne[:, 0], failed_tsne[:, 1], c=failed_labels, cmap=cmap2, alpha=0.7, label='Failure', marker='x')
    # set a separate colorbar for the failed points
    # if the ticks are too many, set the number of ticks to 10
    if np.max(failed_labels) > 10:
        cbar = plt.colorbar(ticks=np.linspace(np.min(failed_labels), np.max(failed_labels), num=10, dtype=int))
        cbar.ax.set_yticklabels([str(i) for i in np.linspace(np.min(failed_labels), np.max(failed_labels), num=10, dtype=int)])
    else:
        cbar = plt.colorbar(ticks=np.arange(np.min(failed_labels), np.max(failed_labels)+1))
        cbar.ax.set_yticklabels([str(i) for i in range(np.min(failed_labels), np.max(failed_labels)+1)])
    cbar.set_label(cbar_label2)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    plt.show()

def visualize_df_tsne(df, title=""):
    """
    Visualize the t-SNE of the successful, failed, successful vs failed episodes in the dataframe by task as well as combined over all tasks
    Args:
        df: pd.DataFrame with columns 'task_name', 'success', 'masked_meanpooled_prefix', 'suffix', 'timestep_number', 'chronological_encoded_subgoals'
        title: title with which to save the plots
    """
    # iterate through each task
    for task in task_list:
        task_path = os.path.join(data_dir, task)
        task_df = df[df['task_name'] == task]
        if task_df.shape[0] == 0:
            continue
        print(f"Processing task: {task}")

        # check if prefix_tsne and suffix_tsne column already exists
        if 'task_prefix_tsne' in task_df.columns and 'task_suffix_tsne' in task_df.columns:
            print(f"t-SNE already computed for task: {task}, skipping...")

        else:
            prefix_tsne = t_SNE(np.vstack(task_df['masked_meanpooled_prefix'].values))
            suffix_tsne = t_SNE(np.vstack(task_df['suffix'].values))
            # add the t-SNE points to the dataframe
            task_df = task_df.reset_index(drop=True)
            # tsne is a np.ndarray of shape (num_steps, 2)
            task_df['task_prefix_tsne'] = list(prefix_tsne)
            task_df['task_suffix_tsne'] = list(suffix_tsne)

        # visualize the prefix successful episode with t-SNE, coloring by timesteps
        success_prefix_df = task_df[task_df['success'] == True]
        visualize_episode_tsne(np.vstack(success_prefix_df['task_prefix_tsne']), np.vstack(success_prefix_df['timestep'], dtype=int), title=f"t-SNE of Successful Prefixes - {task}", save_path=os.path.join(task_path, f'tsne_successful_prefixes_{title}.png'), cbar_label='timestep')

        # visualize the prefix failed episode with t-SNE, coloring by timesteps
        failed_prefix_df = task_df[task_df['success'] == False]
        visualize_episode_tsne(np.vstack(failed_prefix_df['task_prefix_tsne']), np.vstack(failed_prefix_df['timestep'], dtype=int), title=f"t-SNE of Failed Prefixes - {task}", cmap='autumn', marker='x', save_path=os.path.join(task_path, f'tsne_failed_prefixes_{title}.png'), cbar_label='timestep')

        # visualize the prefix successful vs failed episode with t-SNE, coloring by timesteps
        visualize_success_vs_failed_episode_tsne(np.vstack(success_prefix_df['task_prefix_tsne']), np.vstack(success_prefix_df['timestep'], dtype=int), np.vstack(failed_prefix_df['task_prefix_tsne']), np.vstack(failed_prefix_df['timestep'], dtype=int), title=f"t-SNE of Successful vs Failed Prefixes - {task}", save_path=os.path.join(task_path, f'tsne_success_vs_failed_prefixes_{title}.png'), cbar_label1='timestep (success)', cbar_label2='timestep (failure)')

        # visualize the prefix successful episode with t-SNE, coloring by subgoal progress
        visualize_episode_tsne(np.vstack(success_prefix_df['task_prefix_tsne']), np.vstack(success_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful Prefixes - {task}", save_path=os.path.join(task_path, f'tsne_successful_prefixes_subgoals_{title}.png'), cbar_label='Subgoal Progress')

        # visualize the prefix failed episode with t-SNE, coloring by subgoal progress
        failed_prefix_df = task_df[task_df['success'] == False]
        visualize_episode_tsne(np.vstack(failed_prefix_df['task_prefix_tsne']), np.vstack(failed_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Failed Prefixes - {task}", cmap='autumn', marker='x', save_path=os.path.join(task_path, f'tsne_failed_prefixes_subgoals_{title}.png'), cbar_label='Subgoal Progress')

        # visualize the prefix successful vs failed episode with t-SNE, coloring by subgoal progress
        visualize_success_vs_failed_episode_tsne(np.vstack(success_prefix_df['task_prefix_tsne']), np.vstack(success_prefix_df['chronological_encoded_subgoals'], dtype=int), np.vstack(failed_prefix_df['task_prefix_tsne']), np.vstack(failed_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful vs Failed Prefixes - {task}", save_path=os.path.join(task_path, f'tsne_success_vs_failed_prefixes_subgoals_{title}.png'), cbar_label1='Subgoal Progress (success)', cbar_label2='Subgoal Progress (failure)')

        # visualize the suffix successful episode with t-SNE, coloring by timesteps
        success_suffix_df = task_df[task_df['success'] == True]
        visualize_episode_tsne(np.vstack(success_suffix_df['task_suffix_tsne']), np.vstack(success_suffix_df['timestep'], dtype=int), title=f"t-SNE of Successful Suffixes - {task}", save_path=os.path.join(task_path, f'tsne_successful_suffixes_{title}.png'), cbar_label='timestep')

        # visualize the suffix failed episode with t-SNE, coloring by timesteps
        failed_suffix_df = task_df[task_df['success'] == False]
        visualize_episode_tsne(np.vstack(failed_suffix_df['task_suffix_tsne']), np.vstack(failed_suffix_df['timestep'], dtype=int), title=f"t-SNE of Failed Suffixes - {task}", cmap='autumn', marker='x', save_path=os.path.join(task_path, f'tsne_failed_suffixes_{title}.png'), cbar_label='timestep')

        # visualize the suffix successful vs failed episode with t-SNE, coloring by timesteps
        visualize_success_vs_failed_episode_tsne(np.vstack(success_suffix_df['task_suffix_tsne']), np.vstack(success_suffix_df['timestep'], dtype=int), np.vstack(failed_suffix_df['task_suffix_tsne']), np.vstack(failed_suffix_df['timestep'], dtype=int), title=f"t-SNE of Successful vs Failed Suffixes - {task}", save_path=os.path.join(task_path, f'tsne_success_vs_failed_suffixes_{title}.png'), cbar_label1='timestep (success)', cbar_label2='timestep (failure)')

    # visualize all tasks combined
    # check if prefix_tsne and suffix_tsne column already exists
    if 'combined_prefix_tsne' in df.columns and 'combined_suffix_tsne' in df.columns:
        print(f"t-SNE already computed for combined tasks, skipping...")
    else:
        combined_prefix_tsne = t_SNE(np.vstack(df['masked_meanpooled_prefix'].values))
        combined_suffix_tsne = t_SNE(np.vstack(df['suffix'].values))
        # add the t-SNE points to the dataframe
        df = df.reset_index(drop=True)
        df['combined_prefix_tsne'] = list(combined_prefix_tsne)
        df['combined_suffix_tsne'] = list(combined_suffix_tsne)
    
    # visualize the prefix successful episode with t-SNE, coloring by timesteps
    success_prefix_df = df[df['success'] == True]
    visualize_episode_tsne(np.vstack(success_prefix_df['combined_prefix_tsne']), np.vstack(success_prefix_df['timestep'], dtype=int), title=f"t-SNE of Successful Prefixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_successful_prefixes_all_tasks_{title}.png'), cbar_label='timestep')

    # visualize the prefix failed episode with t-SNE, coloring by timesteps
    failed_prefix_df = df[df['success'] == False]
    visualize_episode_tsne(np.vstack(failed_prefix_df['combined_prefix_tsne']), np.vstack(failed_prefix_df['timestep'], dtype=int), title=f"t-SNE of Failed Prefixes - All Tasks", cmap='autumn', marker='x', save_path=os.path.join(data_dir, f'tsne_failed_prefixes_all_tasks_{title}.png'), cbar_label='timestep')

    # visualize the prefix successful vs failed episode with t-SNE, coloring by timesteps
    visualize_success_vs_failed_episode_tsne(np.vstack(success_prefix_df['combined_prefix_tsne']), np.vstack(success_prefix_df['timestep'], dtype=int), np.vstack(failed_prefix_df['combined_prefix_tsne']), np.vstack(failed_prefix_df['timestep'], dtype=int), title=f"t-SNE of Successful vs Failed Prefixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_success_vs_failed_prefixes_all_tasks_{title}.png'), cbar_label1='timestep (success)', cbar_label2='timestep (failure)')

    # visualize the prefix successful episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(np.vstack(success_prefix_df['combined_prefix_tsne']), np.vstack(success_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful Prefixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_successful_prefixes_subgoals_all_tasks_{title}.png'), cbar_label='Subgoal Progress')

    # visualize the prefix failed episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(np.vstack(failed_prefix_df['combined_prefix_tsne']), np.vstack(failed_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Failed Prefixes - All Tasks", cmap='autumn', marker='x', save_path=os.path.join(data_dir, f'tsne_failed_prefixes_subgoals_all_tasks_{title}.png'), cbar_label='Subgoal Progress')

    # visualize the prefix successful vs failed episode with t-SNE, coloring by subgoal progress
    visualize_success_vs_failed_episode_tsne(np.vstack(success_prefix_df['combined_prefix_tsne']), np.vstack(success_prefix_df['chronological_encoded_subgoals'], dtype=int), np.vstack(failed_prefix_df['combined_prefix_tsne']), np.vstack(failed_prefix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful vs Failed Prefixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_success_vs_failed_prefixes_subgoals_all_tasks_{title}.png'), cbar_label1='Subgoal Progress (success)', cbar_label2='Subgoal Progress (failure)')


    # visualize the suffix successful episode with t-SNE, coloring by timesteps
    success_suffix_df = df[df['success'] == True]
    visualize_episode_tsne(np.vstack(success_suffix_df['combined_suffix_tsne']), np.vstack(success_suffix_df['timestep'], dtype=int), title=f"t-SNE of Successful Suffixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_successful_suffixes_all_tasks_{title}.png'), cbar_label='timestep')

    # visualize the suffix failed episode with t-SNE, coloring by timesteps
    failed_suffix_df = df[df['success'] == False]
    visualize_episode_tsne(np.vstack(failed_suffix_df['combined_suffix_tsne']), np.vstack(failed_suffix_df['timestep'], dtype=int), title=f"t-SNE of Failed Suffixes - All Tasks", cmap='autumn', marker='x', save_path=os.path.join(data_dir, f'tsne_failed_suffixes_all_tasks_{title}.png'), cbar_label='timestep')

    # visualize the suffix successful vs failed episode with t-SNE, coloring by timesteps
    visualize_success_vs_failed_episode_tsne(np.vstack(success_suffix_df['combined_suffix_tsne']), np.vstack(success_suffix_df['timestep'], dtype=int), np.vstack(failed_suffix_df['combined_suffix_tsne']), np.vstack(failed_suffix_df['timestep'], dtype=int), title=f"t-SNE of Successful vs Failed Suffixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_success_vs_failed_suffixes_all_tasks_{title}.png'), cbar_label1='timestep (success)', cbar_label2='timestep (failure)')

    # visualize the suffix successful episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(np.vstack(success_suffix_df['combined_suffix_tsne']), np.vstack(success_suffix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful Suffixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_successful_suffixes_subgoals_all_tasks_{title}.png'), cbar_label='Subgoal Progress')

    # visualize the suffix failed episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(np.vstack(failed_suffix_df['combined_suffix_tsne']), np.vstack(failed_suffix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Failed Suffixes - All Tasks", cmap='autumn', marker='x', save_path=os.path.join(data_dir, f'tsne_failed_suffixes_subgoals_all_tasks_{title}.png'), cbar_label='Subgoal Progress')

    # visualize the suffix successful vs failed episode with t-SNE, coloring by subgoal progress
    visualize_success_vs_failed_episode_tsne(np.vstack(success_suffix_df['combined_suffix_tsne']), np.vstack(success_suffix_df['chronological_encoded_subgoals'], dtype=int), np.vstack(failed_suffix_df['combined_suffix_tsne']), np.vstack(failed_suffix_df['chronological_encoded_subgoals'], dtype=int), title=f"t-SNE of Successful vs Failed Suffixes - All Tasks", save_path=os.path.join(data_dir, f'tsne_success_vs_failed_suffixes_subgoals_all_tasks_{title}.png'), cbar_label1='Subgoal Progress (success)', cbar_label2='Subgoal Progress (failure)')



# %% load a few steps to see what the data looks like
# load a task
task = task_list[0]
task_path = os.path.join(data_dir, task)
prefix = load_all_steps(os.path.join(task_path, 'episode_0_success=True'), key='outputs/prefix_out', mask_key='outputs/prefix_mask')


#%% visualize the prefixes by task, colored by timestep and subgoal progress
all_success_prefixes = []
all_success_timesteps = []
all_success_subgoals = []
all_failed_prefixes = []
all_failed_timesteps = []
all_failed_subgoals = []
for task in task_list:
    all_prefixes = []
    all_timesteps = []
    all_subgoals = []
    task_path = os.path.join(data_dir, task)
    if not os.path.isdir(task_path):
        continue
    print(f"Processing task prefix: {task}")
    # load successful episode prefixes
    successful_episode_prefixes = load_and_save_task_episodes_concatenated(task_path, success=True, key='outputs/prefix_out', mask_key='outputs/prefix_mask')
    all_success_prefixes.append(successful_episode_prefixes)

    # load the timesteps for the successful episodes
    successful_timesteps = load_and_save_task_episodes_concatenated(task_path, success=True, key='inputs/timestep')
    all_success_timesteps.append(successful_timesteps)

    # load the subgoals for the successful episodes
    successful_subgoals = load_and_save_task_episodes_concatenated(task_path, success=True, key='inputs/observation/subgoal_successes')
    all_success_subgoals.append(successful_subgoals)

    # load the failed episode prefixes
    failed_episode_prefixes = load_and_save_task_episodes_concatenated(task_path, success=False, key='outputs/prefix_out', mask_key='outputs/prefix_mask')
    
    # load the timesteps for the failed episodes
    failed_timesteps = load_and_save_task_episodes_concatenated(task_path, success=False, key='inputs/timestep')
    
    # load the subgoals for the failed episodes
    failed_subgoals = load_and_save_task_episodes_concatenated(task_path, success=False, key='inputs/observation/subgoal_successes')
    print(failed_subgoals.shape)

    all_prefixes.append(successful_episode_prefixes)
    all_timesteps.append(successful_timesteps)
    all_subgoals.append(successful_subgoals)

    if failed_episode_prefixes.size != 0:
        failed_episode_prefixes = failed_episode_prefixes.reshape(failed_episode_prefixes.shape[0], -1)  # flatten the last two dimensions
        all_failed_prefixes.append(failed_episode_prefixes)
        all_failed_timesteps.append(failed_timesteps)
        all_failed_subgoals.append(failed_subgoals)
        all_prefixes.append(failed_episode_prefixes)
        all_timesteps.append(failed_timesteps)
        all_subgoals.append(failed_subgoals)

    all_prefixes_concat = np.concatenate((all_prefixes), axis=0)
    all_prefixes_concat.shape
    tnse_points = t_SNE(all_prefixes_concat)
    successful_tsne_points = tnse_points[:successful_episode_prefixes.shape[0]]
    failed_tsne_points = tnse_points[successful_episode_prefixes.shape[0]:]

    # visualize the successful episode with t-SNE, coloring by timesteps
    visualize_episode_tsne(successful_tsne_points, successful_timesteps, title=f"t-SNE of successful {task} prefixes", save_path=os.path.join(task_path, 'tsne_successful_prefixes_timesteps.png'), cbar_label='Timestep')

    # visualize the successful episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(successful_tsne_points, successful_subgoals, title=f"t-SNE of successful {task} prefixes", save_path=os.path.join(task_path, 'tsne_successful_prefixes_subgoals.png'), cbar_label='Subgoal Progress')


    if failed_tsne_points.size != 0:
        # visualize the failed episode with t-SNE, coloring by timesteps
        visualize_episode_tsne(failed_tsne_points, failed_timesteps, title=f"t-SNE of failed {task} prefixes", save_path=os.path.join(task_path, 'tsne_failed_prefixes_timesteps.png'), cmap="autumn", marker='x', cbar_label='Timestep')

        # visualize the failed episode with t-SNE, coloring by subgoal progress
        visualize_episode_tsne(failed_tsne_points, failed_subgoals, title=f"t-SNE of failed {task} prefixes", save_path=os.path.join(task_path, 'tsne_failed_prefixes_subgoals.png'), cmap="autumn", marker='x', cbar_label='Subgoal Progress')

        # visualize both successful and failed episodes together with different markers colored by timesteps
        visualize_success_vs_failed_episode_tsne(successful_tsne_points, successful_timesteps, failed_tsne_points, failed_timesteps, title=f"t-SNE of {task} Prefix Success vs Failure", save_path=os.path.join(task_path, 'tsne_success_vs_failure_prefixes_timesteps.png'), cbar_label1='Timestep (Success)', cbar_label2='Timestep (Failure)')

        # visualize both successful and failed episodes together with different markers colored by subgoal progress
        visualize_success_vs_failed_episode_tsne(successful_tsne_points, successful_subgoals, failed_tsne_points, failed_subgoals, title=f"t-SNE of {task} Prefix Success vs Failure", save_path=os.path.join(task_path, 'tsne_success_vs_failure_prefixes_subgoals.png'), cbar_label1='Subgoal Progress (Success)', cbar_label2='Subgoal Progress (Failure)')

#%% visualize all prefixes together
all_successful_prefixes_concat = np.concatenate(all_success_prefixes, axis=0)
all_failed_prefixes_concat = np.concatenate(all_failed_prefixes, axis=0)
all_prefixes = np.concatenate((all_successful_prefixes_concat, all_failed_prefixes_concat), axis=0)
all_tsne_points = t_SNE(all_prefixes)
all_success_tsne_points = all_tsne_points[:all_successful_prefixes_concat.shape[0]]
all_failed_tsne_points = all_tsne_points[all_successful_prefixes_concat.shape[0]:]

# visualize all successful prefixes with t-SNE, coloring by timesteps
all_successful_timesteps_concat = np.concatenate(all_success_timesteps, axis=0)
visualize_episode_tsne(all_success_tsne_points, all_successful_timesteps_concat, title="t-SNE of all successful prefixes", save_path=os.path.join(data_dir, 'tsne_all_successful_prefixes_timesteps.png'), cbar_label='Timestep')

# visualize all successful prefixes with t-SNE, coloring by subgoal progress
all_successful_subgoals_concat = np.concatenate(all_success_subgoals, axis=0)
visualize_episode_tsne(all_success_tsne_points, all_successful_subgoals_concat, title="t-SNE of all successful prefixes", save_path=os.path.join(data_dir, 'tsne_all_successful_prefixes_subgoals.png'), cbar_label='Subgoal Progress')

# visualize all failed prefixes with t-SNE, coloring by timesteps
all_failed_timesteps_concat = np.concatenate(all_failed_timesteps, axis=0)
visualize_episode_tsne(all_failed_tsne_points, all_failed_timesteps_concat, title="t-SNE of all failed prefixes", save_path=os.path.join(data_dir, 'tsne_all_failed_prefixes_timesteps.png'), cbar_label='Timestep')

# visualize all failed prefixes with t-SNE, coloring by subgoal progress
all_failed_subgoals_concat = np.concatenate(all_failed_subgoals, axis=0)
visualize_episode_tsne(all_failed_tsne_points, all_failed_subgoals_concat, title="t-SNE of all failed prefixes", save_path=os.path.join(data_dir, 'tsne_all_failed_prefixes_subgoals.png'), cbar_label='Subgoal Progress')

# visualize both successful and failed prefixes together with different markers colored by timesteps
visualize_success_vs_failed_episode_tsne(all_success_tsne_points, all_successful_timesteps_concat, all_failed_tsne_points, all_failed_timesteps_concat, title="t-SNE of all Prefix Success vs Failure", save_path=os.path.join(data_dir, 'tsne_all_success_vs_failure_prefixes_timesteps.png'), cbar_label1='Timestep (Success)', cbar_label2='Timestep (Failure)')

# visualize both successful and failed prefixes together with different markers colored by subgoal progress
visualize_success_vs_failed_episode_tsne(all_success_tsne_points, all_successful_subgoals_concat, all_failed_tsne_points, all_failed_subgoals_concat, title="t-SNE of all Prefix Success vs Failure", save_path=os.path.join(data_dir, 'tsne_all_success_vs_failure_prefixes_subgoals.png'), cbar_label1='Subgoal Progress (Success)', cbar_label2='Subgoal Progress (Failure)')

#%% visualize the suffixes by task, colored by timestep and subgoal progress
all_successful_suffixes = []
all_successful_timesteps = []
all_successful_subgoals = []
all_failed_suffixes = []
all_failed_timesteps = []
all_failed_subgoals = []
for task in task_list:
    all_suffixes = []
    all_timesteps = []
    all_subgoals = []
    task_path = os.path.join(data_dir, task)
    if not os.path.isdir(task_path):
        continue
    print(f"Processing task: {task}")
    successful_suffixes = load_and_save_task_episodes_concatenated(task_path, success=True, key='outputs/suffix_out_aggregated')
    all_successful_suffixes.append(successful_suffixes)

    successful_timesteps = load_and_save_task_episodes_concatenated(task_path, success=True, key='inputs/timestep')
    all_successful_timesteps.append(successful_timesteps)

    successful_subgoals = load_and_save_task_episodes_concatenated(task_path, success=True, key='inputs/observation/subgoal_successes')
    all_successful_subgoals.append(successful_subgoals)

    failed_suffixes = load_and_save_task_episodes_concatenated(task_path, success=False, key='outputs/suffix_out_aggregated')

    failed_timesteps = load_and_save_task_episodes_concatenated(task_path, success=False, key='inputs/timestep')

    failed_subgoals = load_and_save_task_episodes_concatenated(task_path, success=False, key='inputs/observation/subgoal_successes')

    all_suffixes.append(successful_suffixes)
    all_timesteps.append(successful_timesteps)
    all_subgoals.append(successful_subgoals)

    if failed_suffixes.size != 0:
        all_failed_suffixes.append(failed_suffixes)
        all_failed_timesteps.append(failed_timesteps)
        all_failed_subgoals.append(failed_subgoals)
        all_suffixes.append(failed_suffixes)
        all_timesteps.append(failed_timesteps)
        all_subgoals.append(failed_subgoals)
    

    tnse_points = t_SNE(np.concatenate((all_suffixes), axis=0))
    successful_tsne_points = tnse_points[:successful_suffixes.shape[0]]
    failed_tsne_points = tnse_points[successful_suffixes.shape[0]:]
    # visualize the successful episode with t-SNE, coloring by timesteps
    visualize_episode_tsne(successful_tsne_points, successful_timesteps, title=f"t-SNE of successful {task}", save_path=os.path.join(task_path, 'tsne_successful_suffix_timesteps.png'), cbar_label='Timestep')

    # visualize the successful episode with t-SNE, coloring by subgoal progress
    visualize_episode_tsne(successful_tsne_points, successful_subgoals, title=f"t-SNE of successful {task}", save_path=os.path.join(task_path, 'tsne_successful_suffix_subgoals.png'), cbar_label='Subgoal Progress')

    # visualize the failed episode with t-SNE
    if failed_tsne_points.size != 0:
        visualize_episode_tsne(failed_tsne_points, failed_timesteps, title=f"t-SNE of failed {task}", save_path=os.path.join(task_path, 'tsne_failed_suffix_timesteps.png'), cmap="autumn", marker='x', cbar_label='Timestep')

        # visualize the failed episode with t-SNE, coloring by subgoal progress
        visualize_episode_tsne(failed_tsne_points, failed_subgoals, title=f"t-SNE of failed {task}", save_path=os.path.join(task_path, 'tsne_failed_suffix_subgoals.png'), cmap="autumn", marker='x', cbar_label='Subgoal Progress')

        # visualize both successful and failed episodes together with different markers
        visualize_success_vs_failed_episode_tsne(successful_tsne_points, successful_timesteps, failed_tsne_points, failed_timesteps, title=f"t-SNE of {task} Success vs Failure", save_path=os.path.join(task_path, 'tsne_success_vs_failure_suffix_timesteps.png'), cbar_label1='Timestep (Success)', cbar_label2='Timestep (Failure)')

        # visualize both successful and failed episodes together with different markers colored by subgoal progress
        visualize_success_vs_failed_episode_tsne(successful_tsne_points, successful_subgoals, failed_tsne_points, failed_subgoals, title=f"t-SNE of {task} Success vs Failure", save_path=os.path.join(task_path, 'tsne_success_vs_failure_suffix_subgoals.png'), cbar_label1='Subgoal Progress (Success)', cbar_label2='Subgoal Progress (Failure)')

#%% visualize all suffixes together
all_successful_suffixes_concat = np.concatenate(all_successful_suffixes, axis=0)
all_failed_suffixes_concat = np.concatenate(all_failed_suffixes, axis=0)
all_suffixes = np.concatenate((all_successful_suffixes_concat, all_failed_suffixes_concat), axis=0)

all_successful_timesteps_concat = np.concatenate(all_successful_timesteps, axis=0)
all_failed_timesteps_concat = np.concatenate(all_failed_timesteps, axis=0)
all_timesteps = np.concatenate((all_successful_timesteps_concat, all_failed_timesteps_concat), axis=0)

all_successful_subgoals_concat = np.concatenate(all_successful_subgoals, axis=0)
all_failed_subgoals_concat = np.concatenate(all_failed_subgoals, axis=0)
all_subgoals = np.concatenate((all_successful_subgoals_concat, all_failed_subgoals_concat), axis=0)

all_tsne_points = t_SNE(all_suffixes)
successful_tsne_points = all_tsne_points[:all_successful_suffixes_concat.shape[0]]
failed_tsne_points = all_tsne_points[all_successful_suffixes_concat.shape[0]:]

# visualize the successful episode with t-SNE, coloring by timesteps
visualize_episode_tsne(successful_tsne_points, all_successful_timesteps_concat, title="t-SNE of all successful episodes", save_path=os.path.join(data_dir, 'tsne_all_successful_suffix_timesteps.png'), cbar_label='Timestep')

# visualize the successful episode with t-SNE, coloring by subgoal progress
visualize_episode_tsne(successful_tsne_points, all_successful_subgoals_concat, title="t-SNE of all successful episodes", save_path=os.path.join(data_dir, 'tsne_all_successful_suffix_subgoals.png'), cbar_label='Subgoal Progress')

# visualize the failed episode with t-SNE
visualize_episode_tsne(failed_tsne_points, all_failed_timesteps_concat, title="t-SNE of all failed episodes", save_path=os.path.join(data_dir, 'tsne_all_failed_suffix_timesteps.png'), cmap="autumn", marker='x', cbar_label='Timestep')

# visualize the failed episode with t-SNE, coloring by subgoal progress
visualize_episode_tsne(failed_tsne_points, all_failed_subgoals_concat, title="t-SNE of all failed episodes", save_path=os.path.join(data_dir, 'tsne_all_failed_suffix_subgoals.png'), cmap="autumn", marker='x', cbar_label='Subgoal Progress')

# visualize both successful and failed episodes together with different markers
visualize_success_vs_failed_episode_tsne(successful_tsne_points, all_successful_timesteps_concat, failed_tsne_points, all_failed_timesteps_concat, title="t-SNE of all Success vs Failure", save_path=os.path.join(data_dir, 'tsne_all_success_vs_failure_suffix_timesteps.png'), cbar_label1='Timestep (Success)', cbar_label2='Timestep (Failure)')

# visualize both successful and failed episodes together with different markers colored by subgoal progress
visualize_success_vs_failed_episode_tsne(successful_tsne_points, all_successful_subgoals_concat, failed_tsne_points, all_failed_subgoals_concat, title="t-SNE of all Success vs Failure", save_path=os.path.join(data_dir, 'tsne_all_success_vs_failure_suffix_subgoals.png'), cbar_label1='Subgoal Progress (Success)', cbar_label2='Subgoal Progress (Failure)')


#%% construct a dataframe with the following columns: task name (str), success (bool), episode number (int), timestep number (int), subgoals (list of 0 and 1), masked_meanpooled_prefix (np.array of size (2048,)), suffix (np.array of size (1048,))
records = []
for task in task_list:
    task_path = os.path.join(data_dir, task)
    if not os.path.isdir(task_path):
        continue
    print(f"Processing task for dataframe: {task}")
    
    # sort the episodes by episode number
    episode_list = os.listdir(task_path)
    episode_list = [e for e in episode_list if os.path.isdir(os.path.join(task_path, e))]
    episode_list.sort()  # sort the episode list to get the lowest episode number first

    # iterate through each episode, and then through each step file in the episode
    for episode in episode_list:

        # sort the step files by step number
        episode_path = os.path.join(task_path, episode)
        step_files = [f for f in os.listdir(episode_path) if f.endswith('.npy')]
        step_files.sort()  # sort the step files to get the lowest step number first
        # find the max step number. The step file name has the format step_{step_number}_success={True|False}.npy
        max_step_number = max([int(f.split('_')[1]) for f in step_files])

        for step_file in step_files:
            step_data = np.load(os.path.join(episode_path, step_file), allow_pickle=True).item()
            # apply the mask to the prefix_out
            prefix_mask = step_data['outputs/prefix_mask'].astype(bool)
            prefix_out = step_data['outputs/prefix_out'][prefix_mask]
            # apply mean pooling if the prefix_out. It should be of shape (num_valid_tokens, feature_dim)
            prefix_out = prefix_out.mean(axis=0)
            if isinstance(prefix_out, np.ndarray) and len(prefix_out.shape) > 1:
                prefix_out = prefix_out.mean(axis=0)
            record = {
                'task_name': task,
                'success': 'success=True' in episode,
                'episode_number': int(episode.split('_')[1]),
                'timestep': step_data['inputs/timestep'],
                # progress is the current step number divided by the max step number
                'progress': step_data['inputs/timestep'] / max_step_number,
                'subgoals': step_data['inputs/observation/subgoal_successes'],
                'reverse_encoded_subgoals': encode_subgoals(step_data['inputs/observation/subgoal_successes'], mode='reverse'),
                'chronological_encoded_subgoals': encode_subgoals(step_data['inputs/observation/subgoal_successes'], mode='chronological'),
                'masked_meanpooled_prefix': prefix_out,
                'suffix': step_data['outputs/suffix_out_aggregated']
            }
            records.append(record)

#create a dataframe from the records
df = pd.DataFrame(records)
# save the dataframe for reuse
df.to_pickle(os.path.join(data_dir, 'processed_data.pkl'))
#%%
df.head()
    
#%% visualize prefixes and suffixes with t-SNE
visualize_df_tsne(df)
# %% visualize the last 10% steps with t-SNE
df_last_10 = df[df['progress'] >= 0.9]
visualize_df_tsne(df_last_10, title='last_10%_steps')
#%% visualize the last 25% failed steps with t-SNE
# filter the df for failed last 25% steps and all successful steps
df_failed_last_25 = df[(df['progress'] >= 0.75) & (df['success'] == False) | (df['success'] == True)]
visualize_df_tsne(df_failed_last_25, title='failed_last_25%_steps')
# %%
