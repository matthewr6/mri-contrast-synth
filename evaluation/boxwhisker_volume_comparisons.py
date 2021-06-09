import os
import glob
import numpy as np
import nibabel as nib
import collections

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


paths = [
    'WMn_invwarped',
    'CSFnS_invwarped',
    'CSFnSB_wmn_invwarped',
]

data = {}

compare_mode = 'proportional'
# compare_mode = 'direct'

for predicted_path in paths:
    struct_vol_diffs = collections.defaultdict(list)

    for subj_path in subjs:
        if not os.path.exists(os.path.join(subj_path, 'samseg', original_path, 'samseg.stats')) or not os.path.exists(os.path.join(subj_path, 'samseg', predicted_path, 'samseg.stats')):
            continue
        original_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', original_path))
        predicted_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', predicted_path))
        for structure in original_vols:
            diff = predicted_vols[structure] - original_vols[structure]
            proportional_diff = diff / original_vols[structure]
            if compare_mode == 'direct':
                struct_vol_diffs[structure].append(diff)
            else:
                struct_vol_diffs[structure].append(proportional_diff)

    data[predicted_path] = struct_vol_diffs

structure_groups = [    
    [
        'Left-Lateral-Ventricle',
        'Left-Inf-Lat-Vent',
        'Left-Cerebellum-White-Matter',
        'Left-Cerebellum-Cortex',
        'Left-Thalamus',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Hippocampus',
        'Left-Amygdala',
        'Left-Accumbens-area',
        'Left-VentralDC',
        'Left-choroid-plexus',
    ],
    [
        'Right-Lateral-Ventricle',
        'Right-Inf-Lat-Vent',
        'Right-Cerebellum-White-Matter',
        'Right-Cerebellum-Cortex',
        'Right-Thalamus',
        'Right-Caudate',
        'Right-Putamen',
        'Right-Pallidum',
        'Right-Hippocampus',
        'Right-Amygdala',
        'Right-Accumbens-area',
        'Right-VentralDC',
        'Right-choroid-plexus',
    ],
    [
        '3rd-Ventricle',
        '4th-Ventricle',
        '5th-Ventricle',
    ],
]

group_names = [
    'left_deepbrain',
    'right_deepbrain',
    'asymmetric_deepbrain',
]

colors = [
    '#2CA02C',
    '#FF7F0E',
    '#448DC0',
]
abs_val = (compare_mode == 'direct')

def conv(v):
    if abs_val:
        v = abs(v)
    if compare_mode == 'proportional':
        return v * 100
    return v

for group_idx, structures in enumerate(structure_groups):
    plt.figure()

    idx = 0

    for structure in structures:
        bp = plt.boxplot(
            [[conv(v) for v in data[p][structure]] for p in paths],
            positions=[idx + i + 1 for i in range(len(paths))],
            widths=0.8,
            sym='.',
            showfliers=False,
        )
        for element in ['boxes', 'fliers', 'means', 'medians']:
            for cmp_idx in range(len(paths)):
                if len(bp[element]):
                    plt.setp(bp[element][cmp_idx], color=colors[cmp_idx])
        for element in ['whiskers', 'caps']:
            for cmp_idx in range(len(paths) * 2):
                plt.setp(bp[element][cmp_idx], color=colors[int(cmp_idx / 2)])
        idx += 4

    plt.legend(handles=[mpatches.Patch(color=colors[j], label=l) for j, l in enumerate(paths)])

    plt.xlim(0, idx + 1)
    if compare_mode == 'proportional':
        plt.axhline(xmax=idx + 2, ls='--', color='black')
    # plt.xticks(range(2, len(structures * 4), 4), structures, rotation=90)
    plt.xticks(range(2, len(structures * 4), 4), structures, rotation=45, ha='right')
    if compare_mode == 'direct':
        plt.ylabel('Absolute volume difference (mm^3)')
        plt.yscale('log')
    else:
        plt.ylabel('% volume difference')
    plt.xlabel('Structure')

    plt.tight_layout()

    # plt.show()
    plt.savefig('graphs/boxwhisker_{}_{}.png'.format(compare_mode, group_names[group_idx]))
    plt.clf()
