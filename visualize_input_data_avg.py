import argparse
import os 

from matplotlib import pyplot as plt
import numpy as np
import tables

from config.config_loader import ConfigLoader
from utils_batch import plot_psd


parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str)
parser.add_argument('--run_name', type=str, default='timestamp')
parser.add_argument('--n_samples', type=int, default=70000) # number of samples to plot per stage

args, unknown = parser.parse_known_args()

config = ConfigLoader(run_name=args.run_name, experiment=args.experiment)


class OnlineVariance:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.shape = shape

    def add(self, data):
        # data.shape = batch_size, data_shape
        for x in data:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (x - self.mean)

    def compute(self):
        if self.n < 2:
            return np.zeros(self.shape).astype(np.float32), np.zeros(self.shape).astype(
                np.float32
            )

        variance = self.M2 / (self.n - 1)
        variance = np.reshape(variance, self.shape)
        mean = np.reshape(self.mean, self.shape)

        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
        
with tables.open_file(config.DATA_FILENAME, mode="r") as f:
    table = f.root.merged_datasets
    
    for d_ in np.unique(table.col('dataset')):
        
        d_indexes = table.get_where_list('dataset == {}'.format(d_))
        d_stages = np.unique(table[d_indexes]['label'])
        
        # Create OnlineVariance objects for each prototype and channel
        mean_psd = {}
        for s in d_stages:
            mean_psd[s] = {}  # initialize inner dict
            for c in config.CHANNELS:
                mean_psd[s][c] = OnlineVariance((128,))
        
        for s_ in d_stages:
            s_indexes = table.get_where_list('(dataset == {}) & (label == {})'.format(d_, s_))
            s_indexes = np.random.choice(s_indexes, args.n_samples)
            
            for c in config.CHANNELS:
                samples = table[s_indexes][c]
                mean_psd[s_][c].add(np.squeeze(samples))

        
        fig, axs = plt.subplots(len(d_stages), 1, figsize=(6, 10), dpi=300)
        fig.suptitle('{} - {}'.format(args.experiment, d_.decode('utf-8')))
        colors = plt.cm.viridis(np.linspace(0, 1, len(d_stages)))  # Generate a colormap for d_stages
        for s_idx, (s_, color) in enumerate(zip(d_stages, colors)):
            for c in config.CHANNELS:
                s_mean, s_std = mean_psd[s_][c].compute()
                
                s_mean = 10 * np.log10(np.maximum(s_mean, 1e-12))
                s_std = 10 * np.log10(np.maximum(s_std, 1e-12))
                
                axs[s_idx].plot(s_mean, label=c, color=color)
                # axs[s_idx].fill_between(
                #     np.arange(128),
                #     s_mean - s_std,
                #     s_mean + s_std,
                #     alpha=0.2,
                #     color=color,
                # )
                axs[s_idx].set_title(s_)
                axs[s_idx].set_xlabel("Frequency (Hz)")
                axs[s_idx].set_ylabel("Power (dB)")
                axs[s_idx].set_xticks(np.linspace(0, 127, 8, dtype=int))
                axs[s_idx].set_xticklabels(np.linspace(0, config.SAMPLING_RATE/2, 8, dtype=int))


        plt.tight_layout()
        os.makedirs("./figures", exist_ok=True)
        plt.savefig(f"./figures/{args.experiment}_input_data_{d_.decode('utf-8')}_avg_DB.png")
        plt.close(fig)