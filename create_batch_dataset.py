# Inspired by https://github.com/dslaborg/sleep-mice-tuebingen/blob/master/scripts/transform_files_tuebingen.py

import argparse
import importlib
import time
import os
import tables
import yaml

import numpy as np

from config.config_loader import ConfigLoader

from data_table import (
    create_table_description,
    COLUMN_LABEL,
    COLUMN_SUBJECT_ID,
    COLUMN_SPECIES,
    COLUMN_DATASET
)

# TODO
# create a column for unified labels between humans and mice, and a column with different nrem labels?

subjects_dict = {}


def parse():
    parser = argparse.ArgumentParser(description="data transformation script")
    parser.add_argument(
        "--experiment",
        "-e",
        required=False,
        default="standard_config",
        help="name of experiment to transform data to",
    )

    return parser.parse_args()


def write_data_to_table(
    table: tables.Table, features: dict, labels: list, subject_id: str, species: str, dataset: str
):
    """writes given data to the passed table, each sample is written in a new row"""
    sample = table.row

    # iterate over samples and create rows
    for l_idx, label in enumerate(labels):
        try:
            sample[COLUMN_SUBJECT_ID] = subject_id
            for c in config.CHANNELS:
                if 'EEG' in c:
                    sample[c] = features[c][l_idx]
                    
                sample[c+"_rms"] = features[c+"_rms"][l_idx]
            sample[COLUMN_LABEL] = label
            sample[COLUMN_SPECIES] = species
            sample[COLUMN_DATASET] = dataset
            sample.append()
        except ValueError:
            print(f"""
            Error while processing epoch {l_idx} with label {label}.
            This epoch is ignored.
            """)
    # write data to table
    table.flush()

def transform():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    # check if file already exists and what datasets are present
    if os.path.isfile(config.DATA_FILENAME):
        with tables.open_file(config.DATA_FILENAME, mode="r") as f:
            table = f.root['merged_datasets']
            table_datasets = np.unique(table.col('dataset'))
            table_datasets_set = {s.decode('utf-8') if isinstance(s, bytes) else s for s in table_datasets}

            # Determine which datasets are present and which are not
            datasets_to_table = [s for s in config.DATASETS if s not in table_datasets_set]
            
            if len(datasets_to_table) == 0:
                print(f"{config.DATA_FILENAME} already exists and all datasets in config.DATASETS are present. Exiting.")
                exit()
            else:
                print(f"{config.DATA_FILENAME} already exists with datasets {table_datasets}. Adding {datasets_to_table}.")
            
    else:        
        datasets_to_table = config.DATASETS
        print(f"{config.DATA_FILENAME} does not exist yet. Creating file with datasets: {datasets_to_table}")
        # create h5 file
        with tables.open_file(config.DATA_FILENAME, mode="w") as f:
            # create table
            table = f.create_table(
                f.root, "merged_datasets", table_desc, "all datasets in one table"
            )
    
    with open("data/datasets.yml", "r") as file:
        datasets_config = yaml.safe_load(file)
    
    with tables.open_file(config.DATA_FILENAME, mode="a") as f:
        
        table = f.root['merged_datasets']
        
        for dataset in datasets_to_table:
            # Extract preprocessing module and class name
            preprocessor_path = datasets_config[dataset]["preprocessor"]
            module_name, class_name = preprocessor_path.split(":")  

            # Dynamically import the module
            module = importlib.import_module(module_name)
            Preprocessor = getattr(module, class_name)
            preprocessor = Preprocessor(config, datasets_config[dataset])
            
            recordings_df = preprocessor.get_recordings()
            
            # iterate over files, load them and write them to the created table
            for _, recording in recordings_df.iterrows():
                
                print("processing recording", recording["signal_location"])
                start = time.time()

                features, labels = preprocessor.process_recording(recording["signal_location"],
                                                                  recording["labels_location"])

                write_data_to_table(
                    table,
                    features,
                    labels,
                    dataset + "_" + recording['subject_id'],
                    preprocessor.species,
                    dataset
                )

                # mouse_epochs_counter[m['set']] += len(labels)

                print("execution time: {:.2f}".format(time.time() - start))
                print()

            # print("mouse epochs: ", mouse_epochs_counter)


    # Save subjects_dict, mouse_tasks, and human_tasks
    # with open(
    #     os.path.join(os.path.dirname(config.DATA_FILENAME), "subjects_dict.yaml"), "w"
    # ) as file:
    #     yaml.dump(subjects_dict, file)

    # with open(
    #     os.path.join(os.path.dirname(config.DATA_FILENAME), "mouse_tasks.yaml"), "w"
    # ) as file:
    #     yaml.dump(mouse_tasks, file)

    # with open(
    #     os.path.join(os.path.dirname(config.DATA_FILENAME), "human_tasks.yaml"), "w"
    # ) as file:
    #     yaml.dump(human_tasks, file)


if __name__ == "__main__":
    args = parse()

    config = ConfigLoader(run_name="timestamp", experiment=args.experiment)

    # transform files
    transform()
