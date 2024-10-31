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


def read_architectures(file_path):
    """
    Reads all architectures from the history file and returns them as a list of tuples
    containing (architecture_string, architecture_list).

    Args:
        file_path (str): Path to the architecture history file

    Returns:
        list: List of tuples (architecture_string, architecture_list)
    """
    architectures = []
    current_architecture = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Layer"):
            parts = line.split(":")[1].strip().split()
            if parts[0] == "Not":
                current_architecture.append(None)
            else:
                current_architecture.append(int(parts[0]))
        elif line.startswith("Total") or line.startswith("\n"):
            if (
                current_architecture
            ):  # Only process if we have collected an architecture
                # Create a filename-friendly string representation that includes null layers
                arch_str = "_".join(
                    "n" if x is None else str(x) for x in current_architecture
                )
                architectures.append((arch_str, current_architecture.copy()))
                current_architecture = []  # Reset for next architecture

    return architectures


def format_architecture_for_yaml(architecture):
    """
    Converts an architecture list to a YAML-compatible string.

    Args:
        architecture (list): List of integers/None representing the architecture

    Returns:
        str: YAML-compatible string representation of the architecture
    """
    return str(architecture).replace("None", "null")


def run_fixed_experiment(ex_name, seed, architecture_str, architecture_list):
    """
    Runs a fixed MLP experiment with a given seed and architecture.

    Args:
        seed (int): Seed value for the experiment
        architecture_str (str): String representation of architecture for naming
        architecture_list (list): List of integers/None representing the architecture
    """
    config_path = "/work/inestp02/xipe_markus/self-expanding-neural-networks/senn_mlp/experiment1/default_config.yaml"

    # Modify the seed and architecture
    modify_yaml_line(config_path, "seed", seed)
    modify_yaml_line(
        config_path, "contents", format_architecture_for_yaml(architecture_list)
    )

    # Run the fixed experiment with the specific architecture
    experiment_name = f"{ex_name}.{seed}.{architecture_str}"
    subprocess.run(["python", "experiment1_fixed.py", "--name", experiment_name])


def run_experiments(seed):
    """
    Runs the initial experiment and then trains fixed MLPs for each architecture encountered.

    Args:
        seed (int): The seed value to be used for the experiments
    """
    config_path = "/work/inestp02/xipe_markus/self-expanding-neural-networks/senn_mlp/experiment1/default_config.yaml"
    architecture_path = "/work/inestp02/xipe_markus/self-expanding-neural-networks/senn_mlp/final_architectures/experiment1/"

    # First, run the original expanding network experiment
    modify_yaml_line(config_path, "seed", seed)
    experiment_name = f"ex1.{seed}"
    subprocess.run(["python", "experiment1.py", "--name", experiment_name])

    # Read all architectures from the history file
    architecture_file = f"{architecture_path}{experiment_name}_architecture_history.txt"
    architectures = read_architectures(architecture_file)

    # Run fixed MLP experiments for each unique architecture
    seen_architectures = set()
    for arch_str, arch_list in architectures:
        # Convert to tuple for hashability, keeping None values
        arch_tuple = tuple(arch_list)
        if arch_tuple not in seen_architectures:
            seen_architectures.add(arch_tuple)
            run_fixed_experiment(experiment_name, seed, arch_str, arch_list)

    # Restore the original configuration
    modify_yaml_line(config_path, "seed", 0)
    modify_yaml_line(config_path, "contents", "[1]")


# Run experiments for seeds 0 to 9
def main():
    for seed in range(10, 30):
        print(f"Running experiments with seed {seed}")
        run_experiments(seed)
        print(f"Finished experiments with seed {seed}")

    print("All experiments completed")


if __name__ == "__main__":
    main()
