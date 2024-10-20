import re
import subprocess


def modify_yaml_line(file_path, start_string, new_value):
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
    with open(file_path, "r") as file:
        lines = file.readlines()
    architecture = []
    for line in lines:
        if line.startswith("Layer"):
            neurons = int(line.split(":")[1].strip().split()[0])
            architecture.append(neurons)
    return str(architecture)


def run_experiment(seed):
    config_path = "experiment1/default_config.yaml"

    # Modify the seed
    modify_yaml_line(config_path, "seed", seed)

    # Run the first experiment
    experiment_name = f"ex1.{seed}_test"
    subprocess.run(["python", "experiment1.py", "--name", experiment_name])

    # Read and update the architecture
    architecture_file = f"{experiment_name}_final_architecture.txt"
    new_architecture = read_architecture(architecture_file)
    modify_yaml_line(config_path, "contents", new_architecture)

    # Run the second experiment
    experiment_name = f"ex1_fixed.{seed}"
    subprocess.run(["python", "experiment1_fixed.py", "--name", experiment_name])

    # Restore the original seed and architecture
    modify_yaml_line(config_path, "seed", 0)
    modify_yaml_line(config_path, "contents", "[1]")


# Run experiments for seeds 0 to 9
for seed in range(1):
    print(f"Running experiment with seed {seed}")
    run_experiment(seed)
    print(f"Finished experiment with seed {seed}")

print("All experiments completed")
