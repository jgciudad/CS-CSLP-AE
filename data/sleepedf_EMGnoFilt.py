import os
from typing import Dict, List, Tuple 

import pandas as pd
import numpy as np
import mne
import scipy

from data.base import BasePreprocessor
from data.preprocessing_utils import ButterworthFilter, MultitaperPreprocessor, WelchPreprocessor

class SleepedfPreprocessor(BasePreprocessor):
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config, 'human')
        
        self.annotation_desc_2_event_id = {
            "Sleep stage W": 1,
            "Sleep stage 1": 2,
            "Sleep stage 2": 3,
            "Sleep stage 3": 4,
            "Sleep stage 4": 4,
            "Sleep stage R": 5,
        }
        self.event_id = {
            "WAKE": 1,
            "N1": 2,
            "N2": 3,
            "N3": 4,
            "REM": 5,
        }
        
    def get_recordings(self) -> pd.DataFrame:
        """Finds all .tsv files in self.data_path and extracts the subject ID from the file name.

        Returns:
            pd.DataFrame: DataFrame with columns ['scorer', 'subject_id', 'location']
        """

        filelist = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=list(range(83)), path=self.data_path, on_missing="warn"
        )
        signal_locations = [f[0] for f in filelist]
        label_locations = [f[1] for f in filelist]
        id_list = [f[0].split('/')[-1][3:5] for f in filelist]
        
        df = pd.DataFrame({'subject_id': id_list,
                           'signal_location': signal_locations,
                           'labels_location': label_locations})

        return df
    
    def process_recording(self, signal_path, labels_path) -> Tuple[Dict[str, np.ndarray], list]:
        """reads data from signal_path.edf and labels_path.edf
        
        Code adapted from https://github.com/andersxa/CSLP-AE/blob/main/data_preparation/create_sleepedfx.py

        Returns:
            features_dict: dict where keys are the channels and the values are numpy arrays
            with shape [n_epochs, timepoints, frequency bins].
            labels: sleep stages as a list of strings
        """

        raw = mne.io.read_raw_edf(signal_path, preload=True, stim_channel="Event marker", infer_types=True)
        annot = mne.read_annotations(labels_path)
        raw.set_annotations(annot, emit_warning=False)

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on raw data
        first_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][0]
        last_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][-1]
        if first_wake is not None and last_wake is not None:
            annot.crop(annot[first_wake + 1]["onset"] - 30 * 60, annot[last_wake]["onset"] + 30 * 60)
        else:
            ...
            # continue
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=self.annotation_desc_2_event_id, chunk_duration=30.0)

        # Resample data and events
        raw.resample(self.config.SAMPLING_RATE, events=events, npad="auto")

        tmax = 30.0 - 1.0 / self.config.SAMPLING_RATE
        epochs = mne.Epochs(
            raw=raw, events=events, event_id=self.event_id, tmin=0.0, tmax=tmax, baseline=None, preload=True, on_missing="warn"
        )
        
        features = {}
        for new_c_name, old_c_name in self.dataset_config['channels'].items():
            
            X = epochs.get_data(old_c_name)
            
            if new_c_name == "EEG1" or new_c_name == "EEG2":
                X = self.eeg_filter(X, self.config.SAMPLING_RATE)
                
            # Normalize data
            mu = X.mean()
            std = X.std()
            X = (X - mu) / std
            X = X.squeeze()
            
            if new_c_name == "EEG1" or new_c_name == "EEG2":
                X_rms = np.sqrt(np.mean(X**2, 1))

                if self.config.PREPROCESSING == "multitaper":
                    Preprocessor = MultitaperPreprocessor(**self.config.PREPROCESSING_PARAMS)
                elif self.config.PREPROCESSING == "welch":
                    Preprocessor = WelchPreprocessor(**self.config.PREPROCESSING_PARAMS)
                
                X, freqs = Preprocessor(X)
                
                if self.config.RELATIVE_PSD:
                    total_power = scipy.integrate.simpson(y=X, x=freqs, axis=-1)
                    X = X / total_power[:, np.newaxis]
                
                features[new_c_name] = X
                
            else: 
                if self.config.EMG_MODALITY == "rms":
                    X_rms = np.sqrt(np.mean(X**2, 1))
            
            features[new_c_name+"_rms"] = X_rms

        labels = [list(epochs.event_id.keys())[list(epochs.event_id.values()).index(id)] for id in epochs.events[:, -1].tolist()]
        
        return features, labels
    