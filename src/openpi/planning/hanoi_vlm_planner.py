import os
from typing import *
# add planning to path
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpt import query_gpt

planning_prompt = """You are a helpful robot planner playing towers of hanoi. The differently colored cubes represent the differently sized disks and the three rectangular areas (the left area, the middle area, the right area) represent the three pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube in either an empty area or on top of a cube that is bigger. You should plan a sequence of actions to achieve the goal shown in the second image from the initial state shown in the first image. You are able to execute the following actions: 
pick the {color} cube
place the {color} cube in the {left|middle|right} area. 
place the {color} cube on top of the {color} cube.
Please provide a step-by-step plan where each action step is separated by a newline to achieve the goal. Output your plan in the following example format:
pick the blue cube
place the blue cube in the right area 
"""

def load_image_pairs(path_to_images='/home/hrilab/Documents/.vlas/first_and_last_frames/'):
    """
    Load image pairs from the given path.
    Each image pair consists of an initial state image and a goal state image.
    Args:
        path_to_images (str): Path to the directory containing images.
    Returns:
        list of tuples: Each tuple contains (init_image_path, goal_image_path).
    """
    import os
    image_pairs = []
    for filename in os.listdir(path_to_images):
        if filename.endswith("_init.jpeg"):
            init_image_path = os.path.join(path_to_images, filename)
            goal_image_path = os.path.join(path_to_images, filename.replace("_first_frame.png", "_last_frame.png"))
            if os.path.exists(goal_image_path):
                image_pairs.append((init_image_path, goal_image_path))
    return image_pairs

def query_model(init_image: Union[os.PathLike, np.array], goal_image: Union[os.PathLike, np.array], model='gpt-5') -> List[str]:
    """
    Query the model with the given prompt and images.
    Args:
        prompt (str): The planning prompt.
        init_image (str): Path to the initial state image.
        goal_image (str): Path to the goal state image.
    Returns:
        list of str: The plan as a list of steps.
    """
    return query_gpt(planning_prompt, init_image, goal_image, model=model)
    # return query_qwen(prompt, init_image, goal_image)


if __name__ == "__main__":
    
    pairs = load_image_pairs()
