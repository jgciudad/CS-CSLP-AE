import numpy as np
import torch
from torch.nn import functional as F
from data_table import COLUMN_LABEL, COLUMN_SUBJECT_ID, COLUMN_SPECIES
from matplotlib import pyplot as plt
import wandb

from utils_sleepedfx import plot_psd, plot_spectrogram

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode(True)
def sample_latents(loader, subject_latents, task_latents, target_subject, target_task, n=2000, specific_task='all', specific_subject='all'):
    subjects = loader.table[loader.split_indices][COLUMN_SUBJECT_ID]
    tasks = loader.table[loader.split_indices][COLUMN_LABEL]
    
    # only for debugging
    # subjects = subjects[loader.random_indices]
    # tasks = tasks[loader.random_indices]
    
    if isinstance(specific_subject, int):
        convert_task_latents = task_latents[(subjects == specific_subject) & (tasks == target_task)]
    elif specific_subject == 'all':
        convert_task_latents = task_latents[tasks == target_task]
    elif specific_subject == 'same':
        convert_task_latents = task_latents[(subjects == target_subject) & (tasks == target_task)]
    elif specific_subject == 'different':
        convert_task_latents = task_latents[(subjects != target_subject) & (tasks == target_task)]
    else:
        raise ValueError('specific_subject must be one of [#subject_class_label#, all, same, different]')
    if isinstance(specific_task, int):
        convert_subject_latents = subject_latents[(subjects == target_subject) & (tasks == specific_task)]
    elif specific_task == 'all':
        convert_subject_latents = subject_latents[subjects == target_subject]
    elif specific_task == 'same':
        convert_subject_latents = subject_latents[(subjects == target_subject) & (tasks == target_task)]
    elif specific_task == 'different':
        convert_subject_latents = subject_latents[(subjects == target_subject) & (tasks != target_task)]
    else:
        raise ValueError('specific_task must be one of [#task_class_label#, all, same, different]')
    
    if convert_task_latents.shape[0] != 0:
        num_task_latents = convert_task_latents.shape[0]
        if num_task_latents < n:
            task_permute_idxs = np.random.randint(0, num_task_latents, size=n)
        else:
            task_permute_idxs = np.random.permutation(num_task_latents)[:n]
        convert_task_latents = convert_task_latents[task_permute_idxs]
    else:
        print(f'convert_task_latents is empty with target_subject={target_subject}, target_task={target_task}, specific_subject={specific_subject}, specific_task={specific_task}.')
        print("Ignoring it in reconstruction loss.")

    if convert_subject_latents.shape[0] != 0:
        num_subject_latents = convert_subject_latents.shape[0]
        if num_subject_latents < n:
            subject_permute_idxs = np.random.randint(0, num_subject_latents, size=n)
        else:
            subject_permute_idxs = np.random.permutation(num_subject_latents)[:n]
        convert_subject_latents = convert_subject_latents[subject_permute_idxs]
    else:
        print(f'convert_subject_latents is empty with target_subject={target_subject}, target_task={target_task}, specific_subject={specific_subject}, specific_task={specific_task}.')
        print("Ignoring it in reconstruction loss.")

    return convert_subject_latents, convert_task_latents


@torch.inference_mode(True)
def reconstruct(model, convert_subject_latents, convert_task_latents, batch_size=2048):
    num_latents = convert_subject_latents.shape[0]
    num_batches = int(np.ceil(num_latents / batch_size))
    convert_subject_latents = torch.unflatten(convert_subject_latents.to(device), 1, (model.latent_dim, model.latent_seqs))
    convert_task_latents = torch.unflatten(convert_task_latents.to(device), 1, (model.latent_dim, model.latent_seqs))
    reconstructions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        x_dict = {'s': convert_subject_latents[start_idx:end_idx], 't': convert_task_latents[start_idx:end_idx]}
        x_dict = model.get_x_hat(x_dict)
        reconstructions.append(x_dict['x_hat'])
    return torch.cat(reconstructions, dim=0)

@torch.inference_mode(True)
def get_reconstructed_erps(model, loader, subject_latents, task_latents, t_spec, s_spec, target_subject, target_task1, n):
    
    convert_subject_latents, convert_task_latents = sample_latents(loader, subject_latents, task_latents, target_subject, target_task1, n=n, specific_task=t_spec, specific_subject=s_spec)
    
    if convert_task_latents.shape[0] != 0 and convert_subject_latents.shape[0] != 0:
        reconstructions = reconstruct(model, convert_subject_latents, convert_task_latents)
        # reconstructions = reconstructions*loader.data_std + loader.data_mean # TODO: fix standardization after transformation
        reconstructed_erp1 = reconstructions.mean(0)
    else:
        reconstructed_erp1 = None

    return reconstructed_erp1

@torch.inference_mode(True)
def get_conversion_results(model, loader, subject_latents, task_latents, target_subject, target_task, channel, channel_idx, n, axes=None, plot_fcn=None):
    rows = loader.table.get_where_list('({}=={}) & ({}=={})'.format(COLUMN_SUBJECT_ID, target_subject, COLUMN_LABEL, target_task)).tolist()
    
    real_erp1 = loader.table[rows][channel]
    # real_erp1 = real_erp1.copy()
    real_erp1 = torch.from_numpy(real_erp1).to(device)
    # real_erp1 = real_erp1.to(device) * loader.data_std + loader.data_mean TODO: fix standardization after transformation
    real_erp1 = real_erp1.mean(0)
    recon_erp_ss1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'same', target_subject, target_task, n)
    recon_erp_sd1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'different', target_subject, target_task, n)
    recon_erp_ds1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'same', target_subject, target_task, n)
    recon_erp_dd1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'different', target_subject, target_task, n)
    
    ss_mse = F.mse_loss(recon_erp_ss1[channel_idx], real_erp1.squeeze()) if recon_erp_ss1 is not None else None
    sd_mse = F.mse_loss(recon_erp_sd1[channel_idx], real_erp1.squeeze()) if recon_erp_sd1 is not None else None
    ds_mse = F.mse_loss(recon_erp_ds1[channel_idx], real_erp1.squeeze()) if recon_erp_ds1 is not None else None
    dd_mse = F.mse_loss(recon_erp_dd1[channel_idx], real_erp1.squeeze()) if recon_erp_dd1 is not None else None

    if axes is not None:
        if plot_fcn.__name__ == 'plot_psd':
            plot_fcn(axes[0, target_task-1], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[0, len(loader.unique_tasks) + target_task-1], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[1, target_task-1], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[1, len(loader.unique_tasks) + target_task-1], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)

            if recon_erp_ss1 is not None: 
                plot_fcn(axes[0, target_task-1], recon_erp_ss1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_sd1 is not None:
                plot_fcn(axes[0, len(loader.unique_tasks) + target_task-1], recon_erp_sd1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_ds1 is not None:
                plot_fcn(axes[1, target_task-1], recon_erp_ds1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_dd1 is not None:
                plot_fcn(axes[1, len(loader.unique_tasks) + target_task-1], recon_erp_dd1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
        elif plot_fcn.__name__ == 'plot_spectrogram':
            raise NotImplementedError('Spectrogram transformation plotting is not implemented yet.')
            
    return ss_mse, sd_mse, ds_mse, dd_mse


@torch.inference_mode(True)
def get_full_conversion_results(model, test_loader, subject_latents, task_latents, N, plot, in_channels, channel, channel_idx, figure_results={}):

    ss_mses, sd_mses, ds_mses, dd_mses = [], [], [], []
    mse_results = {}
    for target_subject in test_loader.unique_subjects:
        
        if plot is True:
            input_type = 'psd' # if in_channels == 1 else 'spectrogram' #sticking to PSD for now
            
            if input_type == 'psd':
                fig, axes = plt.subplots(2, 2*len(test_loader.unique_tasks), figsize=(10*len(test_loader.unique_tasks), 10))
                plot_fcn = plot_psd
            elif input_type == 'spectrogram':
                fig, axes = plt.subplots(4, 2*len(test_loader.unique_tasks), figsize=(10,len(test_loader.unique_tasks), 10))
                plot_fcn = plot_spectrogram
            
        for target_task in test_loader.unique_tasks:
            ss_mse, sd_mse, ds_mse, dd_mse = get_conversion_results(model, test_loader, subject_latents, task_latents, target_subject, target_task, channel, channel_idx, N, axes, plot_fcn)
            paradigm_name = test_loader.task_to_label[target_task]
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'] = ss_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'] = sd_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'] = ds_mse
            mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'] = dd_mse
            ss_mses.append(ss_mse)
            sd_mses.append(sd_mse)
            ds_mses.append(ds_mse)
            dd_mses.append(dd_mse)
        
        if plot is True:
            # Add higher-level titles for grouped subplots
            axes[0,0].legend()
            fig.text(0.25, axes[0,0].get_position().y1 + 0.05, "(S.s, S.t)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.75, axes[0,0].get_position().y1 + 0.05, "(S.s, D.t)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.25, axes[1,0].get_position().y1 + 0.05, "(D.s, S.t)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.75, axes[1,0].get_position().y1 + 0.05, "(D.s, D.t)", fontsize=14, ha='center', fontweight='bold')

            figure_results[f'transforms/{target_subject}'] = wandb.Image(fig)
        
    #Calculate per subject mean
    for target_subject in test_loader.unique_subjects:
        ss_results, sd_results, ds_results, dd_results = [], [], [], []
        for target_task in test_loader.unique_tasks:
            paradigm_name = test_loader.task_to_label[target_task]
            ss_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'])
            sd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'])
            ds_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'])
            dd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'])
            
        ss_results = [x for x in ss_results if x is not None]
        sd_results = [x for x in sd_results if x is not None]
        ds_results = [x for x in ds_results if x is not None]
        dd_results = [x for x in dd_results if x is not None]
        
        mse_results[f'MSE/test/mean/{target_subject}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/test/mean/{target_subject}/sd'] = torch.mean(torch.stack(sd_results))
        mse_results[f'MSE/test/mean/{target_subject}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/test/mean/{target_subject}/dd'] = torch.mean(torch.stack(dd_results))
    #Calculate per paradigm mean
    for target_task in test_loader.unique_tasks:
        paradigm_name = test_loader.task_to_label[target_task]
        ss_results, sd_results, ds_results, dd_results = [], [], [], []
        for target_subject in test_loader.unique_subjects:
            ss_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ss'])
            sd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/sd'])
            ds_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/ds'])
            dd_results.append(mse_results[f'MSE/test/{paradigm_name}/{target_subject}/dd'])
        
        ss_results = [x for x in ss_results if x is not None]
        sd_results = [x for x in sd_results if x is not None]
        ds_results = [x for x in ds_results if x is not None]
        dd_results = [x for x in dd_results if x is not None]
        
        mse_results[f'MSE/test/mean/{paradigm_name}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/test/mean/{paradigm_name}/sd'] = torch.mean(torch.stack(sd_results))
        mse_results[f'MSE/test/mean/{paradigm_name}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/test/mean/{paradigm_name}/dd'] = torch.mean(torch.stack(dd_results))
    
    ss_mses = [x for x in ss_mses if x is not None]
    sd_mses = [x for x in sd_mses if x is not None]
    ds_mses = [x for x in ds_mses if x is not None]
    dd_mses = [x for x in dd_mses if x is not None]

    #Calculate overall mean
    mse_results['MSE/test/mean/ss'] = torch.mean(torch.stack(ss_mses))
    mse_results['MSE/test/mean/sd'] = torch.mean(torch.stack(sd_mses))
    mse_results['MSE/test/mean/ds'] = torch.mean(torch.stack(ds_mses))
    mse_results['MSE/test/mean/dd'] = torch.mean(torch.stack(dd_mses))
    
    return figure_results, mse_results