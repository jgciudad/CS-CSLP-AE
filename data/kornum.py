import os
from typing import Dict, List, Tuple 

import pandas as pd
import numpy as np
import mne
import scipy

from data.base import BasePreprocessor
from data.preprocessing_utils import ButterworthFilter, MultitaperPreprocessor, WelchPreprocessor

class KornumPreprocessor(BasePreprocessor):
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config, 'mouse')
        
    def get_recordings(self) -> pd.DataFrame:
        """Finds all .tsv files in self.data_path and extracts the subject ID from the file name.

        Returns:
            pd.DataFrame: DataFrame with columns ['scorer', 'subject_id', 'location']
        """

        scorer_list = []
        id_list = []
        tsv_list = []
        
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".tsv"):
                    scorer_list.append(root.split('/')[-2][11:])
                    id_list.append(file.split('-')[0].upper())
                    tsv_list.append(os.path.join(root, file))
                    
        edf_list = [f.replace('.tsv', '.edf').replace('/tsv', '/EDF') for f in tsv_list]
        
        df = pd.DataFrame({'scorer': scorer_list,
                           'subject_id': id_list,
                           'signal_location': edf_list,
                           'labels_location': tsv_list})

        return df
    
    def process_recording(self, signal_path, labels_path) -> Tuple[Dict[str, np.ndarray], list]:
        """reads data from recording_path.tsv

        Returns:
            features_dict: dict where keys are the channels and the values are numpy arrays
            with shape [n_epochs, timepoints, frequency bins].
            labels: sleep stages as a list of strings
        """

        labels = self.load_labels(labels_path)
        features_dict = self.load_signal(signal_path)

        return features_dict, labels
    
    def load_labels(self, label_path) -> List[str]:
        """
        Load and transform integer labels from a .tsv file to sleep stages names.
        Args:
            label_path (str): The file path to the .tsv file containing the labels.
        Returns:
            List[str]: A list of category names corresponding to the integer labels.
        """
        
        df = pd.read_csv(
            label_path,
            skiprows=9,
            engine="python",
            sep="\t",
            index_col=False
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

        return stages.to_list()
        
    def load_signal(self, signal_path) -> Tuple[Dict[str, np.ndarray], list]:
        """
        Load and preprocess signal data from the given file path.
        Parameters:
        -----------
        signal_path : str
            Path to the signal edf. file.
        Returns:
        --------
        Tuple[Dict[str, np.ndarray], list]
            A tuple containing:
            - A dictionary where keys are channel names (e.g., 'EEG1', 'EEG2', 'EMG') and values are the preprocessed signal data.
            - A list of epoch start times.
        """
        
        raw_data, channels = self.load_resample_edf(signal_path, self.config.SAMPLING_RATE)

        features = {}

        for c, channel in enumerate(channels):
            if channel == "EEG EEG1A-B":
                c_name = "EEG1"
                c_filter = self.eeg_filter
            elif channel == "EEG EEG2A-B":
                c_name = "EEG2"
                c_filter = self.eeg_filter
            elif channel == "EMG EMG":
                c_name = "EMG"
                c_filter = self.emg_filter

            s = c_filter(raw_data[c, :], self.config.SAMPLING_RATE) 

            s = (raw_data[c, :] - np.mean(s, keepdims=True)) / np.std(s, keepdims=True)

            reshaped_s = np.reshape(
                s, (int(s.shape[0] / self.config.SAMPLING_RATE / self.config.EPOCH_LENGTH), -1)
            )
            
            if c_name == "EEG1" or c_name == "EEG2":
                if self.config.PREPROCESSING == "multitaper":
                    Preprocessor = MultitaperPreprocessor(**self.config.PREPROCESSING_PARAMS)
                elif self.config.PREPROCESSING == "welch":
                    Preprocessor = WelchPreprocessor(**self.config.PREPROCESSING_PARAMS)
                    
                X, freqs = Preprocessor(reshaped_s)
                
                if self.config.RELATIVE_PSD:
                    total_power = scipy.integrate.simpson(y=X, x=freqs, axis=-1)
                    X = X / total_power[:, np.newaxis]
                
                features[c_name] = X
                
                X_rms = np.sqrt(np.mean(reshaped_s**2, 1))

            elif c_name == "EMG":
                if self.config.EMG_MODALITY == "average_rms":
                    X_rms = np.sqrt(np.mean(reshaped_s**2, 1))

            features[c_name+"_rms"] = X_rms

        return features
    
    def load_resample_edf(self, file_path, resample_rate=None):
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