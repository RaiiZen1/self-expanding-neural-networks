from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import os
from glob import glob


def export_tensorboard_data(
    logs_base_dir, prefix, tags=None, output_dir="exported_data"
):
    """
    Export TensorBoard tensor data to CSV files from experiment folders matching the prefix.

    Args:
        logs_base_dir (str): Base directory containing experiment folders
        prefix (str): Prefix to filter experiment folders
        tags (list, optional): List of specific tags to export. If None, exports all tensor tags.
        output_dir (str): Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all experiment directories that match the prefix
    experiment_dirs = [
        d for d in glob(os.path.join(logs_base_dir, f"{prefix}*")) if os.path.isdir(d)
    ]

    if not experiment_dirs:
        print(f"No directories found matching prefix '{prefix}' in {logs_base_dir}")
        return

    for exp_dir in experiment_dirs:
        # Get experiment folder name for CSV naming
        folder_name = os.path.basename(exp_dir)
        print(f"\nProcessing experiment: {folder_name}")

        # Find event file in this experiment directory
        event_files = glob(os.path.join(exp_dir, "event*"))

        if not event_files:
            print(f"No event files found in {exp_dir}")
            continue

        for event_file in event_files:
            # Load the event file
            event_acc = EventAccumulator(event_file, size_guidance={"tensors": 0})
            event_acc.Reload()

            # Get list of tags
            available_tags = event_acc.Tags()["tensors"]
            # print(f"Available tags: {available_tags}")
            tags_to_export = tags if tags is not None else available_tags

            # Export each tag's data
            for tag in tags_to_export:
                if tag in available_tags:
                    # Get all tensor events for this tag
                    tensor_events = event_acc.Tensors(tag)

                    # Extract data from tensor events
                    data = []
                    for event in tensor_events:
                        tensor_value = np.frombuffer(
                            event.tensor_proto.tensor_content, dtype=np.float32
                        )
                        data.append(
                            {
                                "wall_time": event.wall_time,
                                "step": event.step,
                                "value": tensor_value.tolist(),
                            }
                        )

                    # Convert to DataFrame
                    df = pd.DataFrame(data)

                    # Generate filename using folder name and tag
                    safe_tag = tag.replace("/", "_").replace("\\", "_")
                    filename = f"{folder_name}_{safe_tag}.csv"
                    filepath = os.path.join(output_dir, filename)

                    # Save to CSV
                    df.to_csv(filepath, index=False)
                    print(f"Exported {tag} to {filepath}")
                else:
                    print(f"Tag {tag} not found in event file")


if __name__ == "__main__":
    # Example usage
    no = 2
    tag = "validation accuracy"
    export_tensorboard_data(
        logs_base_dir="/work/inestp02/xipe_markus/self-expanding-neural-networks/senn_mlp/logs",
        prefix=f"ex{no}",
        tags=[tag],
        output_dir=f"/work/inestp02/xipe_markus/self-expanding-neural-networks/senn_mlp/tensorboard_exports/experiment{no}/{tag}",
    )
