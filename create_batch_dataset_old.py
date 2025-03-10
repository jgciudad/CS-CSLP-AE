# Inspired by https://github.com/dslaborg/sleep-mice-tuebingen/blob/master/scripts/transform_files_tuebingen.py

import argparse
import sys
import time
from glob import glob
import os
from os.path import join, basename, isfile, realpath, dirname
from config.config_loader import ConfigLoader
import h5py
import numpy as np
import tables
import pandas as pd
import mne
import scipy
import yaml
from pathlib import Path
from data.preprocessing_utils import MultitaperPreprocessor, WelchPreprocessor
from sklearn.model_selection import train_test_split


# sys.path.insert(0, realpath(join(dirname(__file__), '..')))

# from base.data.downsample import downsample
# from base.config_loader import ConfigLoader
from data_table import (
    create_table_description,
    COLUMN_LABEL,
    COLUMN_SUBJECT_ID,
    COLUMN_SPECIES,
)

mouse_tasks = {"WAKE": 1, "NREM": 2, "REM": 3, "ARTIFACT": 4}
human_tasks = {"WAKE": 1, "N1": 2, "N2": 2, "N3": 2, "REM": 3, "ARTIFACT": 4}

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


def load_kornum_labels(label_path):
    df = pd.read_csv(
        label_path, skiprows=9, engine="python", sep="\t", index_col=False
    ).iloc[:, 4]  # ParserWarning can be ignored, it is working properly.

    # transform integer labels to category name
    def int_class_to_string(row):
        if row == 1:
            return "WAKE"
        if row == 2:
            return "NREM"
        if row == 3:
            return "REM"
        if (row != 1) & (row != 2) & (row != 3):
            return "ARTIFACT"

    stages = df.apply(int_class_to_string)

    return stages.to_list(), stages.index.values.tolist()


def window_rms(a, window_size, boundary="fill"):
    a2 = np.power(a, 2)
    window = np.ones((1, window_size)) / float(window_size)
    convolved = scipy.signal.convolve2d(a2, window, mode="same", boundary=boundary)
    return np.sqrt(convolved)


def load_resample_edf(file_path, resample_rate=None):
    """
    :param file_path: path to the .edf recording
    :param resample_rate: new sampling rate in Hertz
    :return: numpy array with the signal
    """

    data = mne.io.read_raw_edf(file_path)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    if resample_rate:
        new_num_samples = raw_data.shape[1] / info["sfreq"] * resample_rate
        if new_num_samples.is_integer() is False:
            raise Exception("New number of samples is not integer")

        raw_data = scipy.signal.resample(x=raw_data, num=int(new_num_samples), axis=1)

    return raw_data, channels


def load_raw_kornum_recording(label_path):
    signal_path = join(
        Path(label_path).parents[1],
        label_path.split("/")[0],
        "EDF",
        basename(label_path)[:-4] + ".edf",
    )

    raw_data, channels = load_resample_edf(
        signal_path, resample_rate=config.SAMPLING_RATE
    )

    # Same filters as in nsrr-data.preprocessing.process_functions-process_shss.py. In case of changing the preprocessing, need
    # to change it there too.
    # eeg_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band") TODO uncomment this line and the one below
    # emg_filter = ButterworthFilter(order=4, fc=[10], type="highpass")

    features = {}

    for c, channel in enumerate(channels):
        if channel == "EEG EEG1A-B":
            c_name = "EEG1"
            # c_filter = eeg_filter TODO uncomment
        elif channel == "EEG EEG2A-B":
            c_name = "EEG2"
            # c_filter = eeg_filter TODO uncomment
        elif channel == "EMG EMG":
            c_name = "EMG"
            # c_filter = emg_filter TODO uncomment

        # s = c_filter(raw_data[c, :], config.SAMPLING_RATE) TODO uncomment

        s = raw_data[c, :]  # TODO remove
        s = (s - np.mean(s, keepdims=True)) / np.std(s, keepdims=True)

        reshaped_s = np.reshape(
            s, (int(s.shape[0] / config.SAMPLING_RATE / config.EPOCH_LENGTH), -1)
        )
        
        config.PREPROCESSING_PARAMS["sfreq"] = config.SAMPLING_RATE

        if config.PREPROCESSING == "multitaper":
            Preprocessor = MultitaperPreprocessor(**config.PREPROCESSING_PARAMS)
        elif config.PREPROCESSING == "welch":
            Preprocessor = WelchPreprocessor(**config.PREPROCESSING_PARAMS)
            
        reshaped_s_psd, _ = Preprocessor(reshaped_s)

        if c_name == "EMG":
            if config.EMG_MODALITY == "average_rms":
                reshaped_s_psd = np.tile(
                    np.expand_dims(np.sqrt(np.mean((reshaped_s**2), 1)), 1),
                    [1, reshaped_s_psd.shape[1]],
                )
            elif config.EMG_MODALITY == "moving_rms":
                reshaped_s = window_rms(reshaped_s, config.EMG_RMS_WINDOW)

        features[c_name] = reshaped_s_psd

    sample_start_times = np.arange(
        0,
        features[c_name].shape[0],
        config.SAMPLING_RATE * config.EPOCH_LENGTH,
        dtype=int,
    )

    return features, sample_start_times.tolist()


def read_kornum_recording(label_path: str):
    """reads data from rec_name.edf and rec_name.tsv

    Returns:
        (dict, list, list): features in the form of a dict with entries for each CHANNEL, labels as a list, list of
        start times of samples as indexes in features
    """

    labels, _ = load_kornum_labels(label_path)
    features_dict, _ = load_raw_kornum_recording(label_path)

    return features_dict, labels


def load_spindle_labels(labels_path, scorer):
    df = pd.read_csv(join(config.SPINDLE_DATA_DIR, labels_path), header=None)

    # column names: {1, 2, 3, n, r, w}
    # 1=wake artifact, 2=NREM artifact, 3=REM artifact

    if scorer == 1:
        labels = df[1]
    elif scorer == 2:
        labels = df[2]

    # rename classes and convert class artifacts to unique artifact class
    def rename_class(row):
        if row == "w":
            return "WAKE"
        if row == "n":
            return "NREM"
        if row == "r":
            return "REM"
        if (row != "w") & (row != "n") & (row != "r"):
            return "ARTIFACT"

    stages = labels.apply(rename_class)

    return stages.to_list(), stages.index.values.tolist()


def write_data_to_table(
    table: tables.Table, features: dict, labels: list, subject_id: int, species: str
):
    """writes given data to the passed table, each sample is written in a new row"""
    sample = table.row

    # iterate over samples and create rows
    for l_idx, label in enumerate(labels):
        try:
            sample[COLUMN_SUBJECT_ID] = subject_id
            for c in config.CHANNELS:
                sample[c] = features[c][l_idx]
            sample[COLUMN_LABEL] = label
            sample[COLUMN_SPECIES] = species
            sample.append()
        except ValueError:
            print(f"""
            Error while processing epoch {l_idx} with label {label}.
            This epoch is ignored.
            """)
    # write data to table
    table.flush()


def center_windowing(signal, stages, original_length, new_length):
    n_subepochs = int(
        original_length / new_length
    )  # amount of subepochs in one original epoch

    center_subepochs = np.zeros(
        (
            signal.shape[0] * n_subepochs,
            signal.shape[1],
            config.EPOCH_LENGTH * config.SAMPLING_RATE,
        )
    )  # array to store the subepochs
    new_stages = []

    margin = (
        original_length - n_subepochs * new_length
    ) * config.SAMPLING_RATE  # remaining samples of the original epoch
    assert margin % 1 == 0  # assert margin is integer

    for epoch in range(signal.shape[0]):
        for i in range(n_subepochs):
            subepoch_start = int(
                margin / 2 + i * config.EPOCH_LENGTH * config.SAMPLING_RATE
            )
            subepoch = signal[
                epoch,
                :,
                subepoch_start : subepoch_start
                + int(config.EPOCH_LENGTH * config.SAMPLING_RATE),
            ]
            center_subepochs[epoch * n_subepochs + i] = subepoch
            new_stages.append(stages[epoch])

    return center_subepochs, new_stages


def get_kornum_subjects(path) -> np.ndarray:
    """
    Finds all .edf files in the data folder and extracts the subject ID from the file name.

    Returns:
        np.ndarray: An array of unique subject IDs.
    """

    scorer_list = []
    id_list = []
    location_list = []
    
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".tsv"):
                scorer_list.append(root.split('/')[-2][11:])
                id_list.append(file.split('-')[0].upper())
                location_list.append(os.path.join(root, file))
    
    database = pd.DataFrame({'scorer': scorer_list, 'id': id_list, 'location': location_list})

    return database


def transform():
    """transform files in DATA_DIR to pytables table"""
    # load description of table columns
    table_desc = create_table_description(config)

    # if the transformed data file already exists, ask the user if he wants to overwrite it
    # if isfile(config.DATA_FILENAME):
    #     question = f"{realpath(config.DATA_FILENAME)} already exists, do you want to override? (y/N): "
    #     response = input(question)
    #     if response.lower() != 'y':
    #         exit()

    # open pytables DATA_FILE
    with tables.open_file(config.DATA_FILENAME, mode="w") as f:
        
        # create table
        train_table = f.create_table(
            f.root, "train", table_desc, "cross_species_data"
        )
        valid_table = f.create_table(
            f.root, "validation", table_desc, "cross_species_data"
        )
        test_table = f.create_table(
            f.root, "test", table_desc, "cross_species_data"
        )

        if "kornum" in config.DATASETS:
             
            mouse_df = get_kornum_subjects('/scratch/s202283/data/EEGdata_cleaned')
            unique_ids = mouse_df['id'].unique()
        
            train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
            train_ids, valid_ids = train_test_split(train_ids, test_size=0.15, random_state=42)
            
            mouse_df['set'] = ""
            mouse_df.loc[mouse_df['id'].isin(train_ids), 'set'] = 'train'
            mouse_df.loc[mouse_df['id'].isin(valid_ids), 'set'] = 'valid'
            mouse_df.loc[mouse_df['id'].isin(test_ids), 'set'] = 'test'

            mouse_epochs_counter = {"train": 0, "valid": 0, "test": 0}
            
            # iterate over files, load them and write them to the created table
            for _, m in mouse_df.iterrows():

                start = time.time()

                features, labels = read_kornum_recording(m['location'])

                write_data_to_table(
                    eval(m['set'] + "_table"),
                    features,
                    [mouse_tasks[x] for x in labels],
                    m['id'][1:],
                    "mouse",
                )

                mouse_epochs_counter[m['set']] += len(labels)

                print("execution time: {:.2f}".format(time.time() - start))
                print()

            print("mouse epochs: ", mouse_epochs_counter)

        if "human" in config.SPECIES:
            human_file_list = [
                os.path.join(config.HUMAN_DATA_PATH, file)
                for file in os.listdir(config.HUMAN_DATA_PATH)
                if file.endswith(".h5")
            ]

            human_epochs_counter = {"training": 0, "validation": 0}
            for file_idx, file in enumerate(human_file_list):
                if any(
                    [s in file for s in config.VALIDATION_HUMANS]
                ):  # checks if the human is in the validation set
                    split = "validation"
                else:
                    split = "training"

                if human_epochs_counter[split] < mouse_epochs_counter[split]:
                    print(
                        "human [{:d}/{:d}]: {:s}".format(
                            file_idx + 1,
                            len(human_file_list),
                            os.path.basename(file)[:-3],
                        )
                    )
                    start = time.time()

                    h5_file = h5py.File(file, "r")
                    stages_map = {
                        "0": "WAKE",
                        "1": "N1",
                        "2": "N2",
                        "3": "N3",
                        "4": "REM",
                    }
                    stages = [stages_map[str(s)] for s in h5_file["stages"][()]]
                    signal = np.array(h5_file["data"]["unscaled"])

                    # eeg_filter = ButterworthFilter(order=2, fc=[7, 35], type="band") # REMOVE THIS line, is only for testing with a highpass

                    # for ax0 in range(signal.shape[0]):
                    #     for ax1 in range(signal.shape[1]):
                    #         if ax1 < 2:
                    #             signal[ax0, ax1, :] = eeg_filter(signal[ax0, ax1, :], config.SAMPLING_RATE)

                    signal = (
                        signal - np.mean(signal, axis=(0, 2), keepdims=True)
                    ) / np.std(signal, axis=(0, 2), keepdims=True)

                    subepochs_signals, subepochs_labels = center_windowing(
                        signal, stages, 30, 4
                    )

                    features = {}
                    for c in config.CHANNELS:
                        # no need to filter, the h5 files are already filtered.

                        s = subepochs_signals[
                            :, h5_file["data"]["channel_idx"].attrs[c], :
                        ]

                        if c == "EMG":
                            if config.EMG_MODALITY == "average_rms":
                                s = np.tile(
                                    np.expand_dims(np.sqrt(np.mean((s**2), 1)), 1),
                                    [1, s.shape[1]],
                                )
                            elif config.EMG_MODALITY == "moving_rms":
                                s = window_rms(s, config.EMG_RMS_WINDOW)

                        features[c] = s

                    human_epochs_counter[split] += len(subepochs_labels)

                    write_data_to_table(
                        eval(split + "_table"),
                        features,
                        subepochs_labels,
                        os.path.basename(file)[:-3],
                        "human",
                    )
                else:
                    print(
                        "Same number as mouse epochs reached in split {}".format(split)
                    )
                    print("Mouse epochs: ", mouse_epochs_counter)
                    print("Human epochs: ", human_epochs_counter)

    # Save subjects_dict, mouse_tasks, and human_tasks
    # with open(
    #     os.path.join(os.path.dirname(config.DATA_FILENAME), "subjects_dict.yaml"), "w"
    # ) as file:
    #     yaml.dump(subjects_dict, file)

    with open(
        os.path.join(os.path.dirname(config.DATA_FILENAME), "mouse_tasks.yaml"), "w"
    ) as file:
        yaml.dump(mouse_tasks, file)

    with open(
        os.path.join(os.path.dirname(config.DATA_FILENAME), "human_tasks.yaml"), "w"
    ) as file:
        yaml.dump(human_tasks, file)


if __name__ == "__main__":
    args = parse()

    config = ConfigLoader(run_name="timestamp", experiment=args.experiment)

    # transform files
    transform()
