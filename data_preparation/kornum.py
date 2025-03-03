from mne.time_frequency import psd_array_multitaper
from scipy.signal import welch
import numpy as np


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

        return psd[:, 1:], freqs


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

        return psd[:, 1:], freqs
