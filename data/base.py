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
        
        self.eeg_filter = ButterworthFilter(order=2, fc=config.EEG_HIGHPASS, type="highpass")
        self.emg_filter = ButterworthFilter(order=4, fc=config.EMG_BANDPASS, type="band")


    def get_recordings(self) -> pd.DataFrame:
        '''This method should return a DataFrame with a row per recording, and with 
        at least the following columns: ['subject_id', 'signal_location', 'labels_location']'''
        pass
    
    def process_recording(self) -> Dict[str, np.ndarray]:
        '''This method should return a dictionary where the keys are the channels
        and the values are numpy arrays with shape [n_epochs, timepoints, frequency bins].'''
        pass