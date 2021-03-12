import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

all_paths = [
    glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'),
]

for paths in all_paths:
    processes = []
    running = 0
    process_batch_size = 48
    for subj_path in paths:
        processes.append(
            subprocess.Popen('cd {} && mri_mask WMn.nii.gz CSFn_stripped_registered.nii.gz WMn_stripped.nii.gz'.format(subj_path), shell=True)
        )
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()
