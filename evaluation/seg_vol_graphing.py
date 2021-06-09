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
original_path = 'CSFn_registered'


paths = [
    ('WMn (samseg)', 'WMn'),
    ('Synthesized segmentation', 'synth_seg'),
    ('Synthesized segmentation (weighted)', 'synth_seg_weighted'),
]

data = {}

# metric = 'proportional'
metric = 'vsi'

for _, predicted_path in paths:
    struct_vol_diffs = collections.defaultdict(list)

    for subj_path in subjs:
        if not os.path.exists(os.path.join(subj_path, 'samseg', original_path, 'samseg.stats')) or not os.path.exists(os.path.join(subj_path, 'samseg', predicted_path, 'samseg.stats')):
            continue
        original_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', original_path))
        predicted_vols = samseg_stats_to_dict(os.path.join(subj_path, 'samseg', predicted_path))
        for structure in original_vols:
            if metric == 'proportional':
                diff = predicted_vols[structure] - original_vols[structure]
                metric_val = diff / original_vols[structure]
            else:
                diff = abs(predicted_vols[structure] - original_vols[structure])
                total = predicted_vols[structure] + original_vols[structure]
                metric_val = 1 - (diff / total)
            struct_vol_diffs[structure].append(metric_val)

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

def subcategorybar(X, vals, stdevs=None, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    handles = []
    for i in range(n):
        h = plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge", label=paths[i][0], yerr=None if stdevs is None else stdevs[i])   
        handles.append(h)
    plt.xticks(_X, X, rotation=45, ha='right')
    return handles

for group_idx, structures in enumerate(structure_groups):
    means = []
    stdevs = []
    for _, c in paths:
        submeans = [np.mean(data[c][s]) for s in structures]
        subdevs = [np.std(data[c][s]) for s in structures]
        means.append(submeans)
        stdevs.append(subdevs)

    # stdevs=[None] * len(means)

    plt.figure()
    if metric == 'vsi':
        plt.ylim(0.5, 1)
    handles = subcategorybar(structures, means)#, stdevs)
    plt.legend(handles=handles, loc='lower left')

    if metric == 'Proportional':
        plt.ylabel('Dice score')
    else:
        plt.ylabel('VSI')
    plt.xlabel('Structure')

    plt.tight_layout()

    # plt.show()
    plt.savefig('graphs/bar_volume_{}_{}.png'.format(metric, group_names[group_idx]))
    plt.clf()