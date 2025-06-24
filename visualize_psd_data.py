import os
import argparse

import numpy as np
from matplotlib import pyplot as plt
import tables

from utils_batch_RMS import plot_psd
from config.config_loader import ConfigLoader


parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str)
parser.add_argument('--fold', type=int, default=-1)
parser.add_argument('--n_folds', type=int, default=4)

n_samples = 9

args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    if args.fold == -1:
        FOLD = np.random.randint(0, args.n_folds)
    else:
        FOLD = args.fold
        
    config = ConfigLoader(run_name='timestamp', fold=FOLD, experiment=args.experiment)
    
    with tables.open_file(config.DATA_FILENAME, mode="r") as pytables_file:
        table = pytables_file.root.merged_datasets
        
        for d in config.DATASETS:
            dataset_stages = np.unique(table[table.get_where_list(f'dataset == {repr(d.encode("utf-8"))}')]['label'])

            for c in config.CHANNELS:
                
                if 'EEG' in c:
                    
                    fig, axs = plt.subplots(len(dataset_stages),
                                            n_samples,
                                            figsize=(3.75*n_samples, 1.6*len(dataset_stages)),
                                            dpi=300,
                                            tight_layout=True)

                    for s_idx, s in enumerate(dataset_stages):
                        sample_indices = np.random.choice(np.where(table[table.get_where_list(f'dataset == {repr(d.encode("utf-8"))}')]['label'] == s)[0], n_samples, replace=False)

                        for i_idx, i in enumerate(sample_indices):
                            sample = table[i]
                            
                            x = sample[c]
                            plot_psd(axs[s_idx, i_idx], x.squeeze(), config.SAMPLING_RATE)
                            axs[s_idx, i_idx].set_title(f"{s.decode('utf-8')}")
                    plt.suptitle(f"{d} - {c}")
                    save_dir = os.path.join("data_figures/", args.experiment, f"{d}")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    fig.savefig(os.path.join(save_dir, f"{c}.png"))
                    plt.close(fig)
                                
            
        
        
    