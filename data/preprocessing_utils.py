from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch, butter, sosfiltfilt
import numpy as np
import librosa


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
