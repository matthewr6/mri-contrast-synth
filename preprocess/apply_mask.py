import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

paths = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')

pairs = [
    ('CSFnS_stripped.nii.gz', 'WMn.nii.gz'),
]


for mask_name, infile in pairs:
    outfile = '{}_stripped.nii.gz'.format(infile.replace('_stripped', '').split('.')[0])
    processes = []
    running = 0
    process_batch_size = 48
    for subj_path in paths:
        if os.path.exists(outfile):
            continue
        processes.append(
            subprocess.Popen('cd {} && mri_mask -T 0 {} {} {}'.format(subj_path, infile, mask_name, outfile), shell=True)
        )
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()
