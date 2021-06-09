import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

all_paths = [
    (glob.glob('/data/mradovan/csfn_synthesis/csfn_synthesized/*.nii.gz'), '/data/mradovan/csfn_synthesis/synth_samseg_results'),
]

for paths, outdir in all_paths:
    processes = []
    running = 0
    process_batch_size = 32
    for input_path in paths:
        subj_path = os.path.basename(input_path)
        output_path = os.path.join(outdir, subj_path.split('.')[0])
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