from typing import Dict

import pandas as pd
import numpy as np

from data.preprocessing_utils import ButterworthFilter

class BasePreprocessor():
    def __init__(self, config, dataset_config, species):
        self.config = config
        self.dataset_config = dataset_config
        self.species = species
        self.data_path = dataset_config['data_path']
        
        if isinstance(config.EEG_BANDPASS, (float, int)):
            eeg_type = "highpass"
            print("Highpass filter applied to EEG data with cutoff frequency:", config.EEG_BANDPASS)
        elif isinstance(config.EEG_BANDPASS, (list, tuple)) and len(config.EEG_BANDPASS) == 2:
            eeg_type = "band"
            print("Bandpass filter applied to EEG data with cutoff frequencies:", config.EEG_BANDPASS)
        if isinstance(config.EMG_BANDPASS, (float, int)):
            emg_type = "highpass"
            print("Highpass filter applied to EMG data with cutoff frequency:", config.EMG_BANDPASS)
        elif isinstance(config.EMG_BANDPASS, (list, tuple)) and len(config.EMG_BANDPASS) == 2:
            emg_type = "band"
            print("Bandpass filter applied to EMG data with cutoff frequencies:", config.EMG_BANDPASS)
        print()
        self.eeg_filter = ButterworthFilter(order=15, fc=config.EEG_BANDPASS, type=eeg_type)
        self.emg_filter = ButterworthFilter(order=4, fc=config.EMG_BANDPASS, type=emg_type)

    def get_recordings(self) -> pd.DataFrame:
        '''This method should return a DataFrame with a row per recording, and with 
        at least the following columns: ['subject_id', 'signal_location', 'labels_location']'''
        pass
    
    def process_recording(self) -> Dict[str, np.ndarray]:
        '''This method should return a dictionary where the keys are the channels
        and the values are numpy arrays with shape [n_epochs, timepoints, frequency bins].'''
        pass