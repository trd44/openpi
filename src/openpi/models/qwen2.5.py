from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", 
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

planning_prompt = """You are a helpful robot planner play the game of towers of hanoi. The differently colored cubes represent the disks and the platforms represent the pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube on either an empty platform or on a cube that is bigger. You should plan a sequence of steps to achieve the goal shown in the second image based on the initial state shown in the first image. You are able to execute the following two actions: 
pick up the {cube}
place {cube} on {location}. 
There exists three different sized cubes and three platforms from left to right: platform 1, platform 2, and platform 3.
Please provide a step-by-step plan where each step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
pick up the blue cube
...
"""

# Messages containing multiple images and a text query
def construct_messages(image_path1, image_path2):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{image_path1}"},
                {"type": "image", "image": f"file:///{image_path2}"},
                {"type": "text", "text": planning_prompt},
            ],
        }
    ]


def call_model(messages) -> str:
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=2000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

def process_output(output_text: str) -> list[str]:
    plan = output_text.split("\n")
    plan = [step.strip() for step in plan]
    return plan

if __name__ == "__main__":
    # read the images of initial towers of hanoi states in `/home/train/VLA-probing/openpi/data/hanoi/init`
    init_images_folder = "/home/train/VLA-probing/openpi/data/hanoi/init/"
    init_image_paths = []
    goal_image_path = "/home/train/VLA-probing/openpi/data/hanoi/goal/hanoi_3_cube_goal.jpeg"
    # loop through all images in the init_images_folder
    import os
    plans = []
    for init_image in os.listdir(init_images_folder):
        if init_image.endswith(".jpeg") or init_image.endswith(".jpg") or init_image.endswith(".png"):
            init_image_path = os.path.join(init_images_folder, init_image)
            init_image_paths.append(init_image_path)
    for init_image_path in init_image_paths:
        messages = construct_messages(init_image_path, goal_image_path)
        plan = call_model(messages)
        # plan is now a string, we want to convert it to a list of steps
        plan = process_output(plan)
        print(init_image_path)
        # print the plan step by step separated by newline
        for step in plan:
            print(step)
        print("\n")
        
