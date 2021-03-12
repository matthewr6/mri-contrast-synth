import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

all_paths = [
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_predicted_invwarped.nii.gz'), 'samseg_predicted'),
    # (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn.nii.gz'), 'samseg'),
    (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/WMn.nii.gz'), 'samseg_wmn'),
    (glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*/CSFn_registered.nii.gz'), 'samseg_warped'),
]

for paths, outdir in all_paths:
    processes = []
    running = 0
    process_batch_size = 44
    for input_path in paths:
        subj_dir = os.path.dirname(input_path)
        output_path = os.path.join(subj_dir, outdir)
        if os.path.exists('{}/samseg.stats'.format(output_path)):
            continue
        processes.append(
            subprocess.Popen('samseg --i {} --o {} --threads 16'.format(input_path, output_path), shell=True)
        )
        running += 1
        if running >= process_batch_size:
            running = 0
            for p in processes:
                p.wait()
            processes = []

    for p in processes:
        p.wait()