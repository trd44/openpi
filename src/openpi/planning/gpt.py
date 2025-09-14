import base64
import time
import os
import cv2
from typing import *
import numpy as np
from openai import OpenAI
client = OpenAI()


# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Function to encode the image
# image can be a path or arraylike
def encode_image(image: Union[os.PathLike, np.ndarray]) -> str:
    # if the image is a numpy array, convert it to base64
    if isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode("utf-8")
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def process_output(output_text: str) -> List[str]:
    plan = output_text.split("\n")
    plan = [step.strip() for step in plan]
    return plan

def query_gpt(prompt, init_image: Union[os.PathLike, np.ndarray], goal_image: Union[os.PathLike, np.ndarray], model="gpt-5") -> str:
    # image can be a path or an array
    base64_image_1 = encode_image(init_image)
    base64_image_2 = encode_image(goal_image)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image_1}"
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image_2}"
                    }
                ]
            }
        ]
    )
    out =  response.output_text
    return process_output(out)


if __name__ == "__main__":
    # read the images of initial towers of hanoi states in `/home/train/VLA-probing/openpi/data/hanoi/init`
    init_images_folder = "/home/train/VLA-probing/openpi/data/hanoi/init/"
    init_image_paths = []
    goal_image_path = "/home/train/VLA-probing/openpi/data/hanoi/goal/hanoi_3_cube_goal.jpeg"
    # loop through all images in the init_images_folder
    import os
    inference_times = []
    for init_image in os.listdir(init_images_folder):
        if init_image.endswith(".jpeg") or init_image.endswith(".jpg") or init_image.endswith(".png"):
            init_image_path = os.path.join(init_images_folder, init_image)
            init_image_paths.append(init_image_path)
    for init_image_path in init_image_paths:
        # measure inference time
        start_time = time.time()
        plan = query_gpt(init_image_path, goal_image_path)
        end_time = time.time()
        inference_times.append(end_time - start_time)
        # plan is now a string, we want to convert it to a list of steps
        plan = process_output(plan)
        print(init_image_path)
        # print the plan step by step separated by newline
        for step in plan:
            print(step)
        print("\n")
    print("Average inference time:", sum(inference_times) / len(inference_times), "seconds")