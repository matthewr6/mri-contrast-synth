import os
import glob
import numpy as np
import nibabel as nib
import collections

def samseg_stats_to_dict(path):
    prefix = '# Measure '
    with open(os.path.join(path, 'samseg.stats'), 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        line = line.strip().split(', ')
        structure = line[0].replace(prefix, '')
        ret[structure] = float(line[1])
    return ret

subjs = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')
original_path = 'CSFn'
predicted_path = 'CSFnSB_wmn_invwarped'

struct_vol_diffs = collections.defaultdict(list)

for subj_path in subjs:
    if not os.path.exists(os.path.join(subj_path, 'samseg', predicted_path)):
        continue
    original_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', original_path))
    predicted_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', predicted_path))
    for structure in original_vols:
        diff = predicted_vols[structure] - original_vols[structure]
        proportional_diff = diff / original_vols[structure]
        struct_vol_diffs[structure].append(proportional_diff)

longest = 'Right-Cerebellum-White-Matter'

structures = sorted(struct_vol_diffs.keys())

for structure in structures:
    diffs = struct_vol_diffs[structure]
    padding = ' ' * (len(longest) - len(structure))
    print(structure + padding, str(abs(np.mean(diffs) * 100))[:10], str(np.std(diffs))[:10])
