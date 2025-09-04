from openpi.models.qwen import call_model, construct_messages, process_output

planning_prompt = """You are a helpful robot planner play the game of towers of hanoi. The differently colored cubes represent the disks and the platforms represent the pegs in towers of Hanoi. Smaller cubes are marked with smaller numbers and bigger cubes are marked with bigger numbers. In towers of Hanoi, you can only pick up cubes from the top of the stacks, and you can only place a cube on either an empty platform or on a cube that is bigger. You should plan a sequence of steps to achieve the goal shown in the second image based on the initial state shown in the first image. You are able to execute the following two actions: 
pick up the {cube}
place {cube} on {location}. 
There exists three different sized cubes and three platforms from left to right: platform 1, platform 2, and platform 3.
Please provide a step-by-step plan where each step is separated by a newline to achieve the goal. Let's think step by step. Output your plan in the following example format:
pick up the blue cube
...
"""


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