import collections
import time
from importlib import import_module
import os
from os.path import dirname, join, realpath, exists, normpath
from datetime import datetime
import numpy as np
import yaml
from torch import optim


def update_dict(d_to_update: dict, update: dict):
    """method to update a dict with the entries of another dict

    `d_to_update` is updated by `update`"""
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping) and k != "stages" and k != "preprocessing_params":
            # exception: the whole stages dict is always overwritten so that stages specified in
            # standard_config but not specified in the new config are not included
            d_to_update[k] = update_dict(d_to_update.get(k, {}), v)
        else:
            d_to_update[k] = v
    return d_to_update


class ConfigLoader:
    def __init__(self, run_name, experiment="standard_config"):
        """general class to load config from yaml files into fields of an instance of this class

        first the config from standard_config.yml is loaded and then updated with the entries in the config file
        specified by `experiment`

        for a description of the various configurations see README.md

        Args:
            experiment (str): name of the config file to load without the .yml file extension; file must be in folder
                'config'
            create_dirs (bool): if set to False EXPERIMENT_DIR, MODELS_DIR and VISUALS_DIR are not created
        """
        # base_dir = realpath(join(dirname(__file__), '..'))

        self.experiment = experiment
        config = self.load_config()

        if run_name=='timestamp':
            self.RUN_NAME = experiment + '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.RUN_NAME = experiment + '_' + run_name
        """identifier for the run of the experiment, used in log_file and `VISUALS_DIR`"""

        # general
        self.DEVICE = config["general"]["device"]
        assert self.DEVICE in ["cpu", "cuda"], "DEVICE only support `cpu` or `cuda`"

        # dirs
        self.DATA_FILENAME = join(
            config["dirs"]["new_data_path"], config["dirs"]["new_data_filename"]
        )
        if config["dirs"]["save_results_path"] != '':
            self.SAVE_RESULTS_PATH = join(config["dirs"]["save_results_path"], self.RUN_NAME)
        else:
            self.SAVE_RESULTS_PATH = join(
                'results', self.experiment, self.RUN_NAME
            )
        os.makedirs(self.SAVE_RESULTS_PATH, exist_ok=True)

        # data
        self.DATASETS = config["data"]["datasets"]
        self.CHANNELS = config["data"]["channels"]
        self.SAMPLING_RATE = config["data"]["sampling_rate"]
        assert type(self.SAMPLING_RATE) is int
        self.EPOCH_LENGTH = config["data"]["epoch_length"]
        assert type(self.EPOCH_LENGTH) in [int, float]
        self.DATA_FRACTION = 1.0
        self.SAMPLES_LEFT = 0  # I need to double check the loading of the neighbor epochs in the data loader, it might be wrong
        self.SAMPLES_RIGHT = 0
        self.BATCH_SIZE = 32
        self.STAGE_MAP = config["data"]["stage_map"]
        self.EMG_MODALITY = config["data"]["emg_modality"]
        if self.EMG_MODALITY == "moving_rms":
            try:
                assert "emg_rms_window" in config["data"]
                assert config["data"]["emg_rms_window"]
            except:
                raise Exception(
                    "if 'emg_modality'=='moving_rms', 'emg_rms_window' needs to be specified."
                )
            self.EMG_RMS_WINDOW = config["data"]["emg_rms_window"]
        
        self.PREPROCESSING = config["data"]["preprocessing"]["method"]
        self.RELATIVE_PSD = config["data"]["preprocessing"]["relative_psd"]
        self.PREPROCESSING_PARAMS = config["data"]["preprocessing"]["preprocessing_params"]
        self.PREPROCESSING_PARAMS["sfreq"] = self.SAMPLING_RATE
        self.EEG_HIGHPASS = config["data"]["preprocessing"]["eeg_bandpass"]
        self.EMG_BANDPASS = config["data"]["preprocessing"]["emg_bandpass"]


        # model
        self.HIDDEN_DIMS = config["model"]["hidden_dimensions"]
        self.F1 = config["model"]["F1"]
        self.F2 = config["model"]["F2"]
        self.D = config["model"]["D"]
        self.KERNEL_1 = config["model"]["kernel_1"]
        self.KERNEL_2 = config["model"]["kernel_2"]
        self.DROPOUT = config["model"]["dropout"]
        self.LR = config["model"]["lr"]
        self.BETA = config["model"]["beta"]

        # training
        self.TRAINING_EPOCHS = config["training"]["epochs"]
        self.EARLY_STOPPING = config["training"]["early_stopping"]
        self.EARLY_STOPPING_PATIENCE = (
            config["training"]["early_stopping_patience"]
            / config["training"]["val_check_interval"]
        )  # transformed from number of epochs to number of validation epochs
        self.VAL_CHECK_INTERVAL = config["training"]["val_check_interval"]
        self.WANDB_PROJECT = config["training"]["wandb_project"]
        self.SAMPLES_PER_STAGE = config["training"]["samples_per_stage"]
        self.VALIDATION_CLASSIFIER = config["training"]["validation_classifier"]

        # testing
        self.PLOT_AT_TESTING = config["testing"]["plot_at_testing"]
        self.N_SUBSET = config["testing"]["n_subset"]
        self.N_COMPONENTS = config["testing"]["n_components"]

    def load_config(self):
        """loads config from standard_config.yml and updates it with <experiment>.yml"""
        base_dir = realpath(join(dirname(__file__), "."))
        with open(join(base_dir, "standard_config.yml"), "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
        if self.experiment != "standard_config":
            with open(
                join(base_dir, self.experiment + ".yml"), "r"
            ) as ymlfile:
                config = update_dict(config, yaml.safe_load(ymlfile))
        return config
