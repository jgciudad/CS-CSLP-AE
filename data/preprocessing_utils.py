from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch, butter, sosfiltfilt
import numpy as np


def label_mapping(h5file, config):
    
    table = h5file.root.merged_datasets
    string_labels = table.col('label')  # returns a NumPy array of byte strings
    string_labels = string_labels.astype('U')    
    
    map_func = np.vectorize(lambda x: config.STAGE_MAP[x])
    int_labels = map_func(string_labels).astype(np.int32)
    
    return int_labels
            
def center_windowing(config, signal, stages, original_length, new_length):
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

class ButterworthFilter(object):
    def __init__(self, order: int = 2, fc: float = 100.0, type: str = "lowpass"):
        self.fc = np.asarray(fc)
        self.order = order
        self.type = type
        self.sos = None
        self.Wn = None

    def __call__(self, x: np.ndarray, fs: int):
        self.Wn = 2 * self.fc / fs
        self.sos = butter((self.order), (self.Wn), btype=(self.type), output="sos")
        return sosfiltfilt(self.sos, x)


class MultitaperPreprocessor:
    def __init__(
        self,
        sfreq: float = 128,
        fmin: float = 0.25,
        normalization: str = "full",
        verbose: int = 0,
        bandwidth: float = 1,
    ):
        self.sfreq = sfreq
        self.fmin = fmin
        self.normalization = normalization
        self.verbose = verbose
        self.bandwidth = bandwidth

    def __call__(self, x: np.array):
        psd, freqs = psd_array_multitaper(
            x,
            sfreq=self.sfreq,
            fmin=self.fmin,
            bandwidth=self.bandwidth,
            verbose=self.verbose,
            normalization=self.normalization,
        )
        
        # psd = librosa.power_to_db(psd)

        return psd[:, 1:], freqs, freqs[1:]


class WelchPreprocessor:
    def __init__(
        self,
        sfreq: float = 128,
        nperseg: int = 200,
        noverlap: int = 200 - 15,
        nfft: int = 256,
    ):
        self.sfreq = sfreq
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft


    def __call__(self, x: np.array):
        freqs, psd = welch(
            x=x,
            fs=self.sfreq,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
        )
        
        # psd = librosa.power_to_db(psd)

        return psd[:, np.newaxis, 1:], freqs[1:]


def create_folds(subjects, num_folds=5, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    subjects = np.array(subjects)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_val = int(np.round(val_ratio * n_subjects))
    n_test = int(np.round(test_ratio * n_subjects))

    val_pool = set()
    test_pool = set()

    train_folds = []
    val_folds = []
    test_folds = []

    for fold_idx in range(num_folds):
        # Select validation subjects not already used in validation
        available_for_val = [s for s in subjects if s not in val_pool]
        val_subjects = rng.choice(available_for_val, size=min(n_val, len(available_for_val)), replace=False)
        val_pool.update(val_subjects)

        # Select test subjects not already used in test and not in this fold's val set
        available_for_test = [s for s in subjects if s not in test_pool and s not in val_subjects]
        test_subjects = rng.choice(available_for_test, size=min(n_test, len(available_for_test)), replace=False)
        test_pool.update(test_subjects)

        # Training = all others
        train_subjects = [s for s in subjects if s not in val_subjects and s not in test_subjects]

        # Store fold
        train_folds.append(train_subjects)
        val_folds.append(val_subjects.tolist())
        test_folds.append(test_subjects.tolist())

    return train_folds, val_folds, test_folds
