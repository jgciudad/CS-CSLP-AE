import sys
import wandb
from split_model import SplitLatentModel
import torch
import numpy as np
from utils_batch import get_results, get_split_latents, split_do_tsne, plot_latents, CustomLoader, fit_knn_fn, fit_etc_fn, plot_psd
from conversion_utils_batch import get_full_conversion_results
from config.config_loader import ConfigLoader
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import tables
import os
from data.preprocessing_utils import create_folds

import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument('--experiment', type=str)
parser.add_argument('--fold', type=int)
parser.add_argument('--run_name', type=str)
parser.add_argument('--split', type=str) # train, eval, test
parser.add_argument('--id', type=str, default='')
parser.add_argument('--wandb_directory', type=str, default='converting-erps')
parser.add_argument('--conversion_N', type=int, default=700)
parser.add_argument('--conversion_channel', type=str, default='EEG2')
parser.add_argument('--full_eval', type=int, default=1)
parser.add_argument('--conversion_results', type=int, default=1)
parser.add_argument('--extra_classifiers', type=int, default=1)


args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    config = ConfigLoader(run_name=args.run_name, fold=args.fold, experiment=args.experiment)

    api = wandb.Api()
    
    # Fetch all runs from the project
    runs = api.runs(path=f"{api.viewer.entity}/{args.wandb_directory}")

    # Search for the run with the desired name
    target_run = next((run for run in runs if config.RUN_NAME in run.name), None)

    if target_run:
        print(f"Found run: {target_run.name} with ID: {target_run.id}")
        # Now you can access the run using its ID
        run = api.run(f"{args.wandb_directory}/{target_run.id}")
    else:
        print("WandB run not found.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    SEED = run.config['seed']   
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    IN_CHANNELS = run.config['in_channels']
    NUM_LAYERS = run.config['num_layers']
    KERNEL_SIZE = 4
    CHANNELS = run.config['channels']
    LATENT_DIM = run.config['latent_dim']
    RECON_TYPE = run.config['recon_type']
    TIME_RESOLUTION = run.config['time_resolution']
    FOLD = run.config['fold']
    N_FOLDS = run.config['n_folds']
    
    wandb.init(
            project="converting-erps",
            id=target_run.id,
            resume="allow"
    )
    
    with tables.open_file(config.DATA_FILENAME, mode="r") as pytables_file:
        table = pytables_file.root.merged_datasets
        train_subjects = []
        validation_subjects = []
        test_subjects = []
        
        for d in config.DATASETS:
            dataset_subjects = np.unique(table[table.get_where_list(f'dataset == {repr(d.encode("utf-8"))}')]['subject_id'])
            
            train_folds_d, val_folds_d, test_folds_d = create_folds(dataset_subjects, num_folds=N_FOLDS, seed=42)
            
            train_subjects.extend(train_folds_d[FOLD])
            validation_subjects.extend(val_folds_d[FOLD])
            test_subjects.extend(test_folds_d[FOLD])
        
        if args.split == 'train':
            subjects = train_subjects
        elif args.split == 'eval':
            subjects = validation_subjects
        elif args.split == 'test':
            subjects = test_subjects
            
        with torch.no_grad():
            loader = CustomLoader(config, pytables_file, subjects=subjects, split='dev')

        model = SplitLatentModel(IN_CHANNELS, CHANNELS, LATENT_DIM, NUM_LAYERS, KERNEL_SIZE, recon_type=RECON_TYPE, content_cosine=1, time_resolution=TIME_RESOLUTION)
        model.loader = loader

        losses = run.config['losses']
        
        model.set_losses(
            batch_size=run.config['batch_size'],
            losses=losses,
            loader=loader,
        )
        state_dict = torch.load(os.path.join(config.SAVE_RESULTS_PATH, f"{config.RUN_NAME}.pt"), map_location=device)
        model.load_state_dict(state_dict)
        
        data_out = {}
        model = model.to(device)
        model.eval()
        
        with torch.inference_mode(True):
            model.loader = loader
            model.eval()
            print('Evaluating...', file=sys.stdout, flush=True)
            subject_latents, task_latents, subjects, tasks, runs, losses = get_split_latents(model, loader, loader.get_dataloader(batch_size=model.batch_size, random_sample=False))
            test_results = get_results(subject_latents, task_latents, subjects, tasks, split=args.split, off_class_accuracy=args.full_eval)
            wandb.log(test_results)
            if args.conversion_results:
                figure_results, mse_results = get_full_conversion_results(model, loader, subject_latents, task_latents, args.conversion_N, True, IN_CHANNELS, args.conversion_channel, config.CHANNELS.index(args.conversion_channel))
                wandb.log(mse_results)
            if args.extra_classifiers:
                test_knn_results = get_results(subject_latents, task_latents, subjects, tasks, clf='KNN', fit_clf=fit_knn_fn, split=args.split)
                wandb.log(test_knn_results)
                test_etc_results = get_results(subject_latents, task_latents, subjects, tasks, clf='ETC', fit_clf=fit_etc_fn, split=args.split)
                wandb.log(test_etc_results)
            
            if 'figure_results' not in locals():
                figure_results = {}

            #%%
            print('Reconstructing...', file=sys.stdout, flush=True)
            with torch.no_grad():
                x = model.get_x_hat({})
            #%%
            print('Plotting training reconstructions...', file=sys.stdout, flush=True)
            n_samples = (5, 4)
            fig, axs = plt.subplots(n_samples[0], n_samples[1], figsize=(20, 15))
            for i in range(n_samples[0] * n_samples[1]):
                mse_sample = torch.mean((x['x'][i] - x['x_hat'][i]) ** 2).item()
                plot_psd(axs[i//n_samples[1], i%4], x['x'][i].squeeze().cpu().detach().numpy(), config.SAMPLING_RATE)
                plot_psd(axs[i//n_samples[1], i%4], x['x_hat'][i].squeeze().cpu().detach().numpy(), config.SAMPLING_RATE, f"S{i}, MSE: {mse_sample:.4f}, Stage: {model.loader.task_to_label[x['T'][i].item()]}")
            plt.tight_layout()
            figure_results['results/recon'] = wandb.Image(fig)
            plt.close(fig)
            
            #%%
            print('Running PCA...', file=sys.stdout, flush=True)
            subject_pca = PCA(n_components=2)
            task_pca = PCA(n_components=2)
            
            subject_pca = subject_pca.fit_transform(subject_latents)
            task_pca = task_pca.fit_transform(task_latents)
            #%%
            print('Plotting PCA...', file=sys.stdout, flush=True)
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            
            plot_latents(fig, ax[0], subject_pca, loader, which='subject')
            plot_latents(fig, ax[1], task_pca, loader, which='task')
            ax[0].set_title('Subject PCA')
            ax[1].set_title('Task PCA')
            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()
            figure_results['results/pca'] = wandb.Image(fig)
            plt.close(fig)
            
            #%%
            print('Running TSNE...', file=sys.stdout, flush=True)
            start_time = time.time()
            subject_tsne, task_tsne = split_do_tsne(subject_latents, task_latents)
            end_time = time.time()
            print(f"Time taken to run split_do_tsne: {end_time - start_time} seconds", file=sys.stdout, flush=True)
            #%%
            print('Plotting TSNE...', file=sys.stdout, flush=True)
            fig, ax = plt.subplots(2, 2, figsize=(20, 20))
            plot_latents(fig, ax[0, 0], subject_tsne, loader, which='subject')
            ax[0, 0].set_title('TSNE: Subject latent colored by subject')
            ax[0, 0].legend(ncol=4, markerscale=8)
            plot_latents(fig, ax[1, 0], task_tsne, loader, which='task')
            ax[1, 0].set_title('TSNE: Task latent colored by task')
            ax[1, 0].legend(markerscale=8)
            plot_latents(fig, ax[0, 1], subject_tsne, loader, which='task')
            ax[0, 1].set_title('TSNE: Subject latent colored by task')
            ax[0, 1].legend(markerscale=8)
            plot_latents(fig, ax[1, 1], task_tsne, loader, which='subject')
            ax[1, 1].set_title('TSNE: Task latent colored by subject')
            ax[1, 1].legend(ncol=4, markerscale=8)
            plt.tight_layout()
            figure_results['results/tsne'] = wandb.Image(fig)
            plt.close(fig)
            
            #%%
            #Plot confusion matrix
            print('Plotting confusion matrix...', file=sys.stdout, flush=True)
            subject_cm = test_results['XGB/' + args.split + '/' + 'subject/cm']
            task_cm = test_results['XGB/' + args.split + '/' + 'task/cm']
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            ax = axs.flatten()
            for i, which in enumerate(['subject', 'task']):
                display_labels = []
                if which == 'subject':
                    cm = subject_cm / subject_cm.sum(axis=1, keepdims=True)
                    display_labels = [f'{s}' for s in loader.unique_subjects]
                elif which == 'task':
                    cm = task_cm / task_cm.sum(axis=1, keepdims=True)
                    display_labels = [f'{loader.task_to_label[t]}' for t in loader.unique_tasks]
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                                display_labels=display_labels)
                disp.plot(ax=ax[i], xticks_rotation='vertical', cmap='Blues', values_format='.2f')
                disp.ax_.get_images()[0].set_clim(0, 1)
                if which == 'subject':
                    acc = test_results['XGB/test/subject/balanced_accuracy']
                elif which == 'task':
                    acc = test_results['XGB/test/task/balanced_accuracy']
                ax[i].set_title(f'{which.capitalize()}\nBalanced Accuracy: {100*acc:.2f}%', fontsize=12)
            plt.tight_layout()
            figure_results['results/cm_fig'] = wandb.Image(fig)
            plt.close(fig)
            plt.close('all')
            print('Uploading to wandb', file=sys.stdout, flush=True)
            wandb.log(figure_results)
            print('Done!', file=sys.stdout, flush=True)