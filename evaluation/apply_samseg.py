import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

all_paths = [
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn.nii.gz'), 'samseg'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/WMn.nii.gz'), 'samseg_wmn'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_registered.nii.gz'), 'samseg_warped'),
    
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn.nii.gz'), 'CSFn'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/WMn.nii.gz'), 'WMn'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_registered.nii.gz'), 'CSFn_registered'),
    
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_out_single_slice_invwarped.nii.gz'), 'CSFn_out_single_slice_invwarped'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_out_single_slice_vgg_invwarped.nii.gz'), 'CSFn_out_single_slice_vgg_invwarped'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_out_single_slice_perceptual_invwarped.nii.gz'), 'CSFn_out_single_slice_perceptual_invwarped'),
]

subjs_dir = '/data/mradovan/7T_WMn_3T_CSFn_pairs/*'
volumes = [
    # ('CSFn', ''),
    # ('WMn', ''),
    # ('CSFn_registered', ''),
    # ('CSFnS_invwarped', ''),
    # ('CSFnSB_wmn_invwarped', ''),
    # ('CSFnS', ''),
    # ('WMn_invwarped' ''),
    ('CSFnSB_wmn', ''),
    # ('CSFn_registered', '--save-posteriors'),
    # ('CSFnB_registered', '--save-posteriors'),
]

for volume, opts in volumes:
    processes = []
    running = 0
    process_batch_size = 4
    for subj_dir in glob.glob(subjs_dir):
        input_path = os.path.join(subj_dir, '{}.nii.gz'.format(volume))
        output_path = os.path.join(subj_dir, 'samseg', volume)
        if os.path.exists('{}/samseg.stats'.format(output_path)) and not opts:
            continue
        processes.append(
            subprocess.Popen('samseg --i {} --o {} --threads 16 {}'.format(input_path, output_path, opts), shell=True)
        )   
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()