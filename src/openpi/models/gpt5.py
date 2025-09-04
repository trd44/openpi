import base64
import time
from openai import OpenAI
client = OpenAI()

planning_prompt = """You are a helpful robot planner play the game of towers of hanoi. The differently colored cubes represent the disks and the platforms represent the pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube on either an empty platform or on a cube that is bigger. You should plan a sequence of steps to achieve the goal shown in the second image based on the initial state shown in the first image. You are able to execute the following two actions: 
pick up the {cube}
place {cube} on {location}. 
place {cube} on {cube}.
There exists three different sized cubes and three platforms from left to right: platform 1, platform 2, and platform 3.
Please provide a step-by-step plan where each step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
pick up the blue cube
...
"""

# Function to create a file with the Files API
def create_file(file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_model(image_path_1: str, image_path_2: str) -> str:
    # Example of calling GPT-5 with text and image inputs
    base64_image_1 = encode_image(image_path_1)
    base64_image_2 = encode_image(image_path_2)
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": planning_prompt,
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
    return response.output_text


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
    inference_times = []
    for init_image in os.listdir(init_images_folder):
        if init_image.endswith(".jpeg") or init_image.endswith(".jpg") or init_image.endswith(".png"):
            init_image_path = os.path.join(init_images_folder, init_image)
            init_image_paths.append(init_image_path)
    for init_image_path in init_image_paths:
        # measure inference time
        start_time = time.time()
        plan = call_model(init_image_path, goal_image_path)
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