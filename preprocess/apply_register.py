import os
import sys
import glob
import subprocess
import numpy as np
import nibabel as nib

all_paths = [
    glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'),
]

register_commands = [
    # 'register.sh',
    # 'register_stripped.sh',
    'register_stripped_csfn_only.sh',
]

for paths in all_paths:
    for register_command in register_commands:
        processes = []
        running = 0
        process_batch_size = 2
        for subj_path in paths:
            processes.append(
                subprocess.Popen('cd {} && ~/research/wmn_to_csfn/preprocess/{}'.format(subj_path, register_command), shell=True)
            )
            running += 1
            if running >= process_batch_size:
                running = 0
                for p in processes:
                    p.wait()
                processes = []

        for p in processes:
            p.wait()
