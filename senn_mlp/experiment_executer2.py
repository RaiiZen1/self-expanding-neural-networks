import re
import subprocess


def modify_yaml_line(file_path, start_string, new_value):
    """
    Modifies a specific line in a YAML file that starts with a given string.

    Args:
        file_path (str): The path to the YAML file to be modified.
        start_string (str): The starting string of the line to be modified.
        new_value (str): The new value to replace the existing value in the line.

    Returns:
        None
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith(start_string):
            parts = line.split(":", 1)
            whitespace = parts[0].rstrip(start_string)
            lines[i] = f"{whitespace}{start_string}: {new_value}\n"
            break

    with open(file_path, "w") as file:
        file.writelines(lines)


def read_architecture(file_path):
    """
    Reads a neural network architecture from a file and returns it as a string representation of a list.

    The file should contain lines starting with "Layer" followed by the number of neurons in that layer.
    For example:
    Layer 1: 64 neurons
    Layer 2: 32 neurons

    Args:
        file_path (str): The path to the file containing the architecture description.

    Returns:
        str: A string representation of a list containing the number of neurons in each layer.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    architecture = []
    for line in lines:
        if line.startswith("Layer"):
            neurons = int(line.split(":")[1].strip().split()[0])
            architecture.append(neurons)
    return str(architecture)


def run_experiment(seed):
    """
    Runs a two-phase experiment with a given seed.

    This function performs the following steps:
    1. Modifies the seed in the configuration file.
    2. Runs the first experiment using the modified configuration.
    3. Reads and updates the architecture from the results of the first experiment.
    4. Runs the second experiment using the updated architecture.
    5. Restores the original seed and architecture in the configuration file.

    Args:
        seed (int): The seed value to be used for the experiment.
    """
    config_path = "experiment2/default_config.yaml"

    # Modify the seed
    modify_yaml_line(config_path, "seed", seed)

    # Run the first experiment
    experiment_name = f"ex2.{seed}"
    subprocess.run(["python", "experiment2.py", "--name", experiment_name])

    # Read and update the architecture
    architecture_file = f"{experiment_name}_final_architecture.txt"
    new_architecture = read_architecture(architecture_file)
    modify_yaml_line(config_path, "contents", new_architecture)

    # Run the second experiment
    experiment_name = f"ex2_fixed.{seed}"
    subprocess.run(["python", "experiment2_fixed.py", "--name", experiment_name])

    # Restore the original seed and architecture
    modify_yaml_line(config_path, "seed", 0)
    modify_yaml_line(config_path, "contents", "[null, null, null]")


# Run experiments for seeds 0 to 10
for seed in range(10):
    print(f"Running experiment with seed {seed}")
    run_experiment(seed)
    print(f"Finished experiment with seed {seed}")

print("All experiments completed")
