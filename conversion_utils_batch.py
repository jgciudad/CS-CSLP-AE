import numpy as np
import torch
from torch.nn import functional as F
from data_table import COLUMN_SUBJECT_ID, COLUMN_SPECIES
from matplotlib import pyplot as plt
import wandb

from utils_sleepedfx import plot_psd, plot_spectrogram

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode(True)
def sample_latents(loader, subject_latents, task_latents, target_subject, target_task, n=2000, specific_task='all', specific_subject='all', specific_species='same'):
    '''
        specific_task: whether subject latents come from same or different task
        specific_subject: whether task latents come from same or different subject
        specific_species: whether task latents come from same or different species
    '''
    
    subjects = loader.table[loader.split_indices][COLUMN_SUBJECT_ID]
    tasks = loader.labels_array[loader.split_indices]
    species = loader.table[loader.split_indices][COLUMN_SPECIES]
    
    target_species = loader.table.get_where_list(f'{COLUMN_SUBJECT_ID}=={target_subject}')[0]
    target_species = loader.table[target_species][COLUMN_SPECIES]
    
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
        if specific_species == 'all':
            convert_task_latents = task_latents[(subjects != target_subject) & (tasks == target_task)]
        elif specific_species == 'same':
            convert_task_latents = task_latents[(subjects != target_subject) & (tasks == target_task) & (species == target_species)]
        elif specific_species == 'different':
            convert_task_latents = task_latents[(subjects != target_subject) & (tasks == target_task) & (species != target_species)]
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
        print(f'convert_task_latents is empty with target_subject={target_subject}, target_task={target_task}, specific_subject={specific_subject}, specific_species={specific_species}')
        print("Ignoring it in reconstruction loss.")

    if convert_subject_latents.shape[0] != 0:
        num_subject_latents = convert_subject_latents.shape[0]
        if num_subject_latents < n:
            subject_permute_idxs = np.random.randint(0, num_subject_latents, size=n)
        else:
            subject_permute_idxs = np.random.permutation(num_subject_latents)[:n]
        convert_subject_latents = convert_subject_latents[subject_permute_idxs]
    else:
        print(f'convert_subject_latents is empty with target_subject={target_subject}, target_task={target_task}, specific_task={specific_task}.')
        print("Ignoring it in reconstruction loss.")

    return convert_subject_latents, convert_task_latents


@torch.inference_mode(True)
def reconstruct(model, convert_subject_latents, convert_task_latents, batch_size=2048):
    num_latents = convert_subject_latents.shape[0]
    num_batches = int(np.ceil(num_latents / batch_size))
    convert_subject_latents = torch.unflatten(torch.tensor(convert_subject_latents, device=device), 1, (model.latent_dim, model.latent_seqs))
    convert_task_latents = torch.unflatten(torch.tensor(convert_task_latents, device=device), 1, (model.latent_dim, model.latent_seqs))
    reconstructions = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i+1) * batch_size
        x_dict = {'s': convert_subject_latents[start_idx:end_idx], 't': convert_task_latents[start_idx:end_idx]}
        x_dict = model.get_x_hat(x_dict)
        reconstructions.append(x_dict['x_hat'])
    return torch.cat(reconstructions, dim=0)

@torch.inference_mode(True)
def get_reconstructed_erps(model, loader, subject_latents, task_latents, t_spec, s_spec, sp_spec, target_subject, target_task1, n):
    
    convert_subject_latents, convert_task_latents = sample_latents(loader, subject_latents, task_latents, target_subject, target_task1, n=n, specific_task=t_spec, specific_subject=s_spec, specific_species=sp_spec)
    
    if convert_task_latents.shape[0] != 0 and convert_subject_latents.shape[0] != 0:
        reconstructions = reconstruct(model, convert_subject_latents, convert_task_latents)
        # reconstructions = reconstructions*loader.data_std + loader.data_mean # TODO: fix standardization after transformation
        reconstructed_erp1 = reconstructions.mean(0)
    else:
        reconstructed_erp1 = None

    return reconstructed_erp1

@torch.inference_mode(True)
def get_conversion_results(model, loader, subject_latents, task_latents, target_subject, target_task, channel, channel_idx, n, axes=None, plot_fcn=None):
    subject_rows = loader.table.col(COLUMN_SUBJECT_ID) == target_subject # no need to check split, subjects are only in one split
    task_rows = loader.labels_array == target_task
    rows = subject_rows & task_rows
    
    real_erp1 = np.ascontiguousarray(loader.table[rows][channel])
    # real_erp1 = real_erp1.copy()
    real_erp1 = torch.from_numpy(real_erp1.copy()).to(device)
    # real_erp1 = real_erp1.to(device) * loader.data_std + loader.data_mean TODO: fix standardization after transformation
    real_erp1 = real_erp1.mean(0)
    recon_erp_ss1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'same', 'same', target_subject, target_task, n)
    recon_erp_ds1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'same', 'same', target_subject, target_task, n)
    recon_erp_sds1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'different', 'same', target_subject, target_task, n)
    recon_erp_dds1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'different', 'same', target_subject, target_task, n)
    recon_erp_sdd1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'same', 'different', 'different', target_subject, target_task, n)
    recon_erp_ddd1 = get_reconstructed_erps(model, loader, subject_latents, task_latents, 'different', 'different', 'different', target_subject, target_task, n)
    
    
    ss_mse = F.mse_loss(recon_erp_ss1[channel_idx], real_erp1.squeeze()) if recon_erp_ss1 is not None else None
    ds_mse = F.mse_loss(recon_erp_ds1[channel_idx], real_erp1.squeeze()) if recon_erp_ds1 is not None else None
    sds_mse = F.mse_loss(recon_erp_sds1[channel_idx], real_erp1.squeeze()) if recon_erp_sds1 is not None else None
    dds_mse = F.mse_loss(recon_erp_dds1[channel_idx], real_erp1.squeeze()) if recon_erp_dds1 is not None else None
    sdd_mse = F.mse_loss(recon_erp_sdd1[channel_idx], real_erp1.squeeze()) if recon_erp_sdd1 is not None else None
    ddd_mse = F.mse_loss(recon_erp_ddd1[channel_idx], real_erp1.squeeze()) if recon_erp_ddd1 is not None else None

    if axes is not None:
        if plot_fcn.__name__ == 'plot_psd':
            plot_fcn(axes[0, target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[0, len(loader.unique_tasks) + target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[1, target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            plot_fcn(axes[1, len(loader.unique_tasks) + target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
            
            if axes.shape[0] > 2:
                plot_fcn(axes[2, target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)
                plot_fcn(axes[2, len(loader.unique_tasks) + target_task], real_erp1.squeeze().cpu().numpy(), label='Real', fs=100)

            if recon_erp_ss1 is not None: 
                plot_fcn(axes[0, target_task], recon_erp_ss1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_ds1 is not None:
                plot_fcn(axes[0, len(loader.unique_tasks) + target_task], recon_erp_ds1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_sds1 is not None:
                plot_fcn(axes[1, target_task], recon_erp_sds1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_dds1 is not None:
                plot_fcn(axes[1, len(loader.unique_tasks) + target_task], recon_erp_dds1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_sdd1 is not None:
                plot_fcn(axes[2, target_task], recon_erp_sdd1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            if recon_erp_ddd1 is not None:
                plot_fcn(axes[2, len(loader.unique_tasks) + target_task], recon_erp_ddd1[channel_idx].cpu().numpy(), title=loader.task_to_label[target_task], label='Trans.', fs=100)
            
        
        elif plot_fcn.__name__ == 'plot_spectrogram':
            raise NotImplementedError('Spectrogram transformation plotting is not implemented yet.')
            
    return ss_mse, ds_mse, sds_mse, dds_mse, sdd_mse, ddd_mse


@torch.inference_mode(True)
def get_full_conversion_results(model, test_loader, subject_latents, task_latents, N, plot, in_channels, channel, channel_idx, split, figure_results={}):
    '''
    Code of losses:
     - first letter indicates whether the subject embedding comes from the same or different task.
     - second letter indicates whether the task embedding comes from the same or different subject
     - third letter indicates whether the task embedding comes from the same or different species (only relevant if 2nd letter is d)
     
    Examples:
     - sdd means that the subject embedding comes from the same task, and the task embedding comes from a different subject and species.
     - dds means that the subject embedding comes from a different task, and the task embedding comes from a different subject but same species.
    '''
    
    ss_mses, ds_mses, sds_mses, dds_mses, sdd_mses, ddd_mses = [], [], [], [], [], []
    mse_results = {}
    
    for target_subject in test_loader.unique_subjects:
        
        if plot is True:
            input_type = 'psd' # if in_channels == 1 else 'spectrogram' #sticking to PSD for now
            
            if input_type == 'psd':
                fig, axes = plt.subplots(test_loader.species.size + 1, 2*len(test_loader.unique_tasks), figsize=(10*len(test_loader.unique_tasks), 10))
                plot_fcn = plot_psd
            elif input_type == 'spectrogram':
                fig, axes = plt.subplots(4, 2*len(test_loader.unique_tasks), figsize=(10,len(test_loader.unique_tasks), 10))
                plot_fcn = plot_spectrogram
            
        for target_task in test_loader.unique_tasks:
            ss_mse, ds_mse, sds_mse, dds_mse, sdd_mse, ddd_mse = get_conversion_results(model, test_loader, subject_latents, task_latents, target_subject, target_task, channel, channel_idx, N, axes, plot_fcn)
            paradigm_name = test_loader.task_to_label[target_task]
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ss'] = ss_mse
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ds'] = ds_mse
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sds'] = sds_mse
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/dds'] = dds_mse
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sdd'] = sdd_mse
            mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ddd'] = ddd_mse
            ss_mses.append(ss_mse)
            ds_mses.append(ds_mse)
            sds_mses.append(sds_mse)
            dds_mses.append(dds_mse)
            sdd_mses.append(sdd_mse)
            ddd_mses.append(ddd_mse)
        
        if plot is True:
            # Add higher-level titles for grouped subplots
            axes[0,0].legend()
            fig.text(0.25, axes[0,0].get_position().y1 + 0.05, "(S.t, S.s)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.75, axes[0,0].get_position().y1 + 0.05, "(D.t, S.s)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.25, axes[1,0].get_position().y1 + 0.05, "(S.t, D.s, S.sp)", fontsize=14, ha='center', fontweight='bold')
            fig.text(0.75, axes[1,0].get_position().y1 + 0.05, "(D.t, D.s, S.sp)", fontsize=14, ha='center', fontweight='bold')
            
            if axes.shape[0] > 2:
                fig.text(0.25, axes[2,0].get_position().y1 + 0.05, "(S.t, D.s, D.sp)", fontsize=14, ha='center', fontweight='bold')
                fig.text(0.75, axes[2,0].get_position().y1 + 0.05, "(D.t, D.s, D.sp)", fontsize=14, ha='center', fontweight='bold')

            figure_results[f'transforms/{split}/{target_subject}'] = wandb.Image(fig)
        
    #Calculate per subject mean
    for target_subject in test_loader.unique_subjects:
        ss_results, ds_results, sds_results, dds_results, sdd_results, ddd_results = [], [], [], [], [], []
        for target_task in test_loader.unique_tasks:
            paradigm_name = test_loader.task_to_label[target_task]
            ss_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ss'])
            ds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ds'])
            sds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sds'])
            dds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/dds'])
            sdd_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sdd'])
            ddd_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ddd'])
            
        ss_results = [x for x in ss_results if x is not None]
        ds_results = [x for x in ds_results if x is not None]
        sds_results = [x for x in sds_results if x is not None]
        dds_results = [x for x in dds_results if x is not None]
        sdd_results = [x for x in sdd_results if x is not None]
        ddd_results = [x for x in ddd_results if x is not None]
        
        mse_results[f'MSE/{split}/mean/{target_subject}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/{split}/mean/{target_subject}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/{split}/mean/{target_subject}/sds'] = torch.mean(torch.stack(sds_results))
        mse_results[f'MSE/{split}/mean/{target_subject}/dds'] = torch.mean(torch.stack(dds_results))
        mse_results[f'MSE/{split}/mean/{target_subject}/sdd'] = torch.mean(torch.stack(sdd_results)) if sdd_results else None
        mse_results[f'MSE/{split}/mean/{target_subject}/ddd'] = torch.mean(torch.stack(ddd_results)) if ddd_results else None
        
    #Calculate per paradigm mean
    for target_task in test_loader.unique_tasks:
        paradigm_name = test_loader.task_to_label[target_task]
        ss_results, ds_results, sds_results, dds_results, sdd_results, ddd_results = [], [], [], [], [], []
        for target_subject in test_loader.unique_subjects:
            ss_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ss'])
            ds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ds'])
            sds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sds'])
            dds_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/dds'])
            sdd_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/sdd'])
            ddd_results.append(mse_results[f'MSE/{split}/{paradigm_name}/{target_subject}/ddd'])
        
        ss_results = [x for x in ss_results if x is not None]
        ds_results = [x for x in ds_results if x is not None]
        sds_results = [x for x in sds_results if x is not None]
        dds_results = [x for x in dds_results if x is not None]
        sdd_results = [x for x in sdd_results if x is not None]
        ddd_results = [x for x in ddd_results if x is not None]
        
        mse_results[f'MSE/{split}/mean/{paradigm_name}/ss'] = torch.mean(torch.stack(ss_results))
        mse_results[f'MSE/{split}/mean/{paradigm_name}/ds'] = torch.mean(torch.stack(ds_results))
        mse_results[f'MSE/{split}/mean/{paradigm_name}/sds'] = torch.mean(torch.stack(sds_results))
        mse_results[f'MSE/{split}/mean/{paradigm_name}/dds'] = torch.mean(torch.stack(dds_results))
        mse_results[f'MSE/{split}/mean/{paradigm_name}/sdd'] = torch.mean(torch.stack(sdd_results)) if sdd_results else None
        mse_results[f'MSE/{split}/mean/{paradigm_name}/ddd'] = torch.mean(torch.stack(ddd_results)) if ddd_results else None
    
    ss_mses = [x for x in ss_mses if x is not None]
    ds_mses = [x for x in ds_mses if x is not None]
    sds_mses = [x for x in sds_mses if x is not None]
    dds_mses = [x for x in dds_mses if x is not None]
    sdd_mses = [x for x in sdd_mses if x is not None]
    ddd_mses = [x for x in ddd_mses if x is not None]

    #Calculate overall mean
    mse_results[f'MSE/{split}/mean/ss'] = torch.mean(torch.stack(ss_mses))
    mse_results[f'MSE/{split}/mean/ds'] = torch.mean(torch.stack(ds_mses))
    mse_results[f'MSE/{split}/mean/sds'] = torch.mean(torch.stack(sds_mses))
    mse_results[f'MSE/{split}/mean/dds'] = torch.mean(torch.stack(dds_mses))
    mse_results[f'MSE/{split}/mean/sdd'] = torch.mean(torch.stack(sdd_mses)) if sdd_mses else None
    mse_results[f'MSE/{split}/mean/ddd'] = torch.mean(torch.stack(ddd_mses)) if ddd_mses else None
    
    return figure_results, mse_results