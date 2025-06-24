import pandas as pd
import mne

from data.sleepedf import SleepedfPreprocessor

class SleepedfDiscardPreprocessor(SleepedfPreprocessor):
    '''
    Preprocessor for Sleep-EDF dataset that discards specific subjects because they lack epochs of one or more stages,
    which leads to issues with pair sampling in the quadruplet loss function.
    This class inherits from SleepedfPreprocessor and overrides the get_recordings method
    '''
    def __init__(self, config, dataset_config):
        super().__init__(config, dataset_config)
        
    def get_recordings(self) -> pd.DataFrame:
        """Finds all .tsv files in self.data_path and extracts the subject ID from the file name.

        Returns:
            pd.DataFrame: DataFrame with columns ['scorer', 'subject_id', 'location']
        """

        filelist = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=list(range(83)), path=self.data_path, on_missing="warn"
        )
        
        signal_locations = []
        label_locations = []
        id_list = []
        
        for f in filelist:
            f_subject_id = f[0].split('/')[-1][3:5]
            if f_subject_id not in ["33", "73", "64", "72", "74", "34"]:
                signal_locations.append(f[0])
                label_locations.append(f[1])
                id_list.append(f_subject_id)
        
        df = pd.DataFrame({'subject_id': id_list,
                           'signal_location': signal_locations,
                           'labels_location': label_locations})

        return df
    
    