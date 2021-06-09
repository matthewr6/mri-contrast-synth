import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

base_path = '/data/mradovan/7T_WMn_3T_CSFn_pairs/*'

all_paths = [
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'), 'CSFn_out_single_slice.nii.gz'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'), 'CSFn_out_single_slice_vgg.nii.gz'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'), 'CSFn_out_single_slice_perceptual.nii.gz'),
]

volumes = [
    # ('CSFnS', 'output'),
    # ('CSFnSB_wmn', 'stripped_wmn_output'),
    # ('WMn', 'output'),
    # ('WMnB_direct', 'output'),
    ('synth_seg', 'output'),
    ('synth_seg_weighted', 'output'),
]

for volume, warp_name in volumes:
    processes = []
    running = 0
    process_batch_size = 32
    for subj_path in glob.glob(base_path):
        if os.path.exists(os.path.join(subj_path, '{}_invwarped.nii.gz'.format(volume))):
            continue
        processes.append(
            subprocess.Popen('cd {subj_path} && WarpImageMultiTransform 3 {infile}.nii.gz {infile}_invwarped.nii.gz -R CSFn.nii.gz -i {warp_name}0GenericAffine.mat {warp_name}1InverseWarp.nii.gz'.format(subj_path=subj_path, infile=volume, warp_name=warp_name), shell=True)
        )
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()
