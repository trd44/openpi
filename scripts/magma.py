import torch
from PIL import Image
from io import BytesIO
import requests
import os
import sys
import json
import torch
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import VisualizationWrapper
import robosuite as suite

from transformers import AutoModelForCausalLM, AutoProcessor
import mimicgen

# Add libero to the path
sys.path.append('/home/train/VLA-probing/openpi/third_party/libero_original')
sys.path.append('/home/train/VLA-probing/Magma/agents/mimicgen')


# Create a mimicgen suite environment
env = suite.make(
    env_name="RedBlueBlocks",
    has_renderer=False,  # Set to False initially to avoid display issues
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_heights=224,
    camera_widths=224,
    robots="Panda",
    controller_configs=load_controller_config(default_controller="OSC_POSE"),
)

# Reset the environment
obs = env.reset()
img = obs["agentview_image"]

step = 0
replay_images = []

dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to("cuda")


convs = [
    {"role": "system", "content": "You are an agent that can see, talk and act."},
    {"role": "user", "content": "<image_start><image><image_end>\nDo you see a red block?"},
    {"role": "assistant", "content": "yes, there is a red block in the image."},
    {"role": "user", "content": "Where is it?"},
    # {"role": "assistant", "content": "The red block is in the air, flying above a blue block."},
    # #{"role": "user", "content": "<image_start><image><image_end>\nPoint to the red block"},
    # {"role": "user", "content": "What is it like to see a red block?"},
    # {"role": "assistant", "content": "Seeing a red block is visually striking, as it stands out against the background. The bright color and distinct shape of the block can draw attention and create a sense of contrast with other objects in the scene."}
    #  Unusual output
    # It is an interesting and unexpected sight to see a red block, especially when it appears to be flying through the air. The contrast between the red block and the blue block, as well as the fact that the red block is not typically associated with blocks, can create a visually engaging and thought-provoking scene. The image of the two blocks, one red and one blue, floating in the air can evoke curiosity and wonder about the nature of these objects and their behavior."}
]
prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
inputs = processor(images=[img], texts=prompt, return_tensors="pt")
inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
inputs = inputs.to("cuda").to(dtype)

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 1.0, 
    "do_sample": False, 
    "use_cache": True,
    "num_beams": 1,
}

with torch.inference_mode():
    generate_ids = model.generate(**inputs, **generation_args)

generate_ids = generate_ids[:, inputs["input_ids"].shape[-1] :]
response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
print(response)
