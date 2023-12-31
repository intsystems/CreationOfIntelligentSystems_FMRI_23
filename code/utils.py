#preprocessing from other group
import os
import numpy as np
import nibabel as nib
import torch
from multiprocessing import Process, Queue
import pandas as pd


def read_hcp(file_path, global_norm_path, per_voxel_norm_path, count, queue=None):
    img_orig = torch.from_numpy(np.asanyarray(nib.load(file_path).dataobj)).to(dtype=torch.float32)
    background = img_orig == 0
    img_temp = (img_orig - img_orig[~background].mean()) / (img_orig[~background].std())
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3)
    #if len(os.listdir(global_norm_path)) != 641:
    for i, TR in enumerate(img):
        TR[TR.isnan()] = 0.0
        torch.save(TR.clone(),
                  os.path.join(global_norm_path, 'rfMRI_' + '_TR_' + str(i) + '.pt'))
    #if len(os.listdir(per_voxel_norm_path)) != 641:
    # repeat for per voxel normalization
    img_temp = (img_orig - img_orig.mean(dim=3, keepdims=True)) / (img_orig.std(dim=3, keepdims=True))
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]

    img[img.isnan()] = 0.0
    img[img.isinf()] = 0.0
    img = torch.split(img, 1, 3)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                  os.path.join(per_voxel_norm_path, 'rfMRI_' + '_TR_' + str(i) + '.pt'))
    print('finished another subject. count is now {}'.format(count))

def preprocess(subjects_dir = 'ds003688-download'):

    subjects = pd.read_csv(subjects_dir + 'participants.tsv', sep='\t')
    subjects_fmri = subjects[subjects['fMRI'] == 'yes']
    queue = Queue()
    count = 0
    for subj_id in subjects_fmri['participant_id']:
        print(subj_id)
        if subj_id not in ['sub-04', 'sub-07']:
            continue
        if len(os.listdir(subjects_dir + f'/fMRI_to_tensors/{subj_id}/glob_norm')) == 641:
            if len(os.listdir(subjects_dir + f'/fMRI_to_tensors/{subj_id}/vox_norm')) == 641:
                continue

        fmri_path = os.path.join(subjects_dir, subj_id, 'ses-mri3t', 'func', f'{subj_id}_ses-mri3t_task-film_run-1_bold.nii.gz')
        if not os.path.exists(fmri_path):
            continue
        glob_norm_path = os.path.join(subjects_dir, 'fMRI_to_tensors', subj_id, 'glob_norm')
        vox_norm_path = os.path.join(subjects_dir, 'fMRI_to_tensors', subj_id, 'vox_norm')
        os.makedirs(glob_norm_path, exist_ok=True)
        os.makedirs(vox_norm_path, exist_ok=True)
        count += 1
        print('start working on subject '+ subj_id)

        #read_hcp(fmri_path, glob_norm_path, vox_norm_path)
        p = Process(target=read_hcp, args=(fmri_path, glob_norm_path, vox_norm_path, count, queue))
        p.start()
        if count % 2 == 0:
            p.join()  # this blocks until the process terminates