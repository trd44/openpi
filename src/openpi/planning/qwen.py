from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time

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


# Messages containing multiple images and a text query
def construct_messages(prompt, init_image_path, goal_image_path):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file:///{init_image_path}"},
                {"type": "image", "image": f"file:///{goal_image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

def process_output(output_text: str) -> list[str]:
    plan = output_text.split("\n")
    plan = [step.strip() for step in plan]
    return plan


def query_qwen(prompt, init_image, goal_image) -> str:
    # Preparation for inference
    messages = construct_messages(prompt, init_image, goal_image)
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
    return process_output(output_text)


if __name__ == "__main__":
    # read the images of initial towers of hanoi states in `/home/train/VLA-probing/openpi/data/hanoi/init`
    init_images_folder = "/home/train/VLA-probing/openpi/data/hanoi/init/"
    init_image_paths = []
    goal_image_path = "/home/train/VLA-probing/openpi/data/hanoi/goal/hanoi_3_cube_goal.jpeg"
    # loop through all images in the init_images_folder
    import os
    plans = []
    inference_times = []
    for init_image in os.listdir(init_images_folder):
        if init_image.endswith(".jpeg") or init_image.endswith(".jpg") or init_image.endswith(".png"):
            init_image_path = os.path.join(init_images_folder, init_image)
            init_image_paths.append(init_image_path)
    for init_image_path in init_image_paths:
        messages = construct_messages(init_image_path, goal_image_path)
        start_time = time.time()
        plan = query_qwen(messages)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        # plan is now a string, we want to convert it to a list of steps
        plan = process_output(plan)
        print(init_image_path)
        # print the plan step by step separated by newline
        for step in plan:
            print(step)
        print("\n")
    print("Average inference time:", sum(inference_times) / len(inference_times), "seconds")
        
