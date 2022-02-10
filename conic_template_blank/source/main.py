

import torch


def run(
        input_dir: str,
        output_dir: str,
        user_data_dir: str,
    ) -> None:
    """Entry function for automatic evaluation.

    This is the function which will be called by the organizer
    docker template to trigger evaluation run. All the data
    to be evaluated will be provided in "input_dir" while
    all the results that will be measured must be saved
    under "output_dir". Participant auxiliary data is provided
    under  "user_data_dir".

    input_dir (str): Path to the directory which contains input data.
    output_dir (str): Path to the directory which will contain output data.
    user_data_dir (str): Path to the directory which contains user data. This
        data include model weights, normalization matrix etc. .

    """
    # ===== Header script for user checking
    print(f"INPUT_DIR: {input_dir}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"CUDA: {torch.cuda.is_available()}")
    for device in range(torch.cuda.device_count()):
        print(f"---Device {device}: {torch.cuda.get_device_name(0)}")
    print("USER_DATA_DIR: ", os.listdir(user_data_dir))

    # ===== Whatever you need
