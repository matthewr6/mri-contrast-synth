import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

paths = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')

infiles = [
    # 'CSFn.nii.gz',
    # 'CSFnS.nii.gz',
    'WMn.nii.gz',
]

for infile in infiles:
    if 'WMn' in infile:
        outfile = '{}B_direct.nii.gz'.format(infile.split('.')[0])
    else:
        outfile = '{}B.nii.gz'.format(infile.split('.')[0])
    processes = []
    running = 0
    process_batch_size = 48
    for subj_path in paths:
        processes.append(
            subprocess.Popen('cd {} && mri_watershed {} {}'.format(subj_path, infile, outfile), shell=True)
        )
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()
