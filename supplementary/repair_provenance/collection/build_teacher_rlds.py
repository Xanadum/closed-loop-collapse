"""Build the SimEggplantTeacher RLDS dataset into ./tfds_data (Phase B, step 1)."""

import os

import tensorflow_datasets as tfds

# importing the module registers the builder with TFDS
import sim_eggplant_teacher_dataset_builder  # noqa: F401

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfds_data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    builder = tfds.builder("sim_eggplant_teacher", data_dir=DATA_DIR)
    builder.download_and_prepare()
    ds_info = builder.info
    print("=== BUILD DONE ===")
    print("data_dir:", DATA_DIR)
    print("splits:", {k: v.num_examples for k, v in ds_info.splits.items()})
    print("features:", ds_info.features)


if __name__ == "__main__":
    main()
