import os
import pickle
import sys
import glob
from tqdm import tqdm
import numpy as np
import nibabel as nib
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

structures = [
    (0, 'Unknown'),
    (2, 'Left-Cerebral-White-Matter'),
    (3, 'Left-Cerebral-Cortex'),
    (4, 'Left-Lateral-Ventricle'),
    (5, 'Left-Inf-Lat-Vent'),
    (7, 'Left-Cerebellum-White-Matter'),
    (8, 'Left-Cerebellum-Cortex'),
    (10, 'Left-Thalamus'),
    (11, 'Left-Caudate'),
    (12, 'Left-Putamen'),
    (13, 'Left-Pallidum'),
    (14, '3rd-Ventricle'),
    (15, '4th-Ventricle'),
    (16, 'Brain-Stem'),
    (17, 'Left-Hippocampus'),
    (18, 'Left-Amygdala'),
    (24, 'CSF'),
    (26, 'Left-Accumbens-area'),
    (28, 'Left-VentralDC'),
    (30, 'Left-vessel'),
    (31, 'Left-choroid-plexus'),
    (41, 'Right-Cerebral-White-Matter'),
    (42, 'Right-Cerebral-Cortex'),
    (43, 'Right-Lateral-Ventricle'),
    (44, 'Right-Inf-Lat-Vent'),
    (46, 'Right-Cerebellum-White-Matter'),
    (47, 'Right-Cerebellum-Cortex'),
    (49, 'Right-Thalamus'),
    (50, 'Right-Caudate'),
    (51, 'Right-Putamen'),
    (52, 'Right-Pallidum'),
    (53, 'Right-Hippocampus'),
    (54, 'Right-Amygdala'),
    (58, 'Right-Accumbens-area'),
    (60, 'Right-VentralDC'),
    (62, 'Right-vessel'),
    (63, 'Right-choroid-plexus'),
    (72, '5th-Ventricle'),
    (77, 'WM-hypointensities'),
    (80, 'non-WM-hypointensities'),
    (85, 'Optic-Chiasm'),
    (165, 'Skull'),
    (258, 'Head-ExtraCerebral'),
    (259, 'Eye-Fluid'),
]

struct_dice_scores = {}
for _, struct in structures:
    struct_dice_scores[struct] = []

def dice_score(seg_a, seg_b, seg_idx):
    X = (seg_a == seg_idx).astype(bool)
    Y = (seg_b == seg_idx).astype(bool)
    X_intersect_Y = X & Y
    return 2 * X_intersect_Y.sum() / (X.sum() + Y.sum())

subjs = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')

# original_path = 'samseg/CSFn/seg.mgz'
# original_path_basename = 'CSFn'

original_path = 'samseg/CSFn_registered/seg.mgz'
original_path_basename = 'CSFn_registered'

predicted_paths = [
    # ('WMn_invwarped', 'samseg/WMn_invwarped/seg.mgz'),
    # ('CSFnSB_wmn_invwarped', 'samseg/CSFnSB_wmn_invwarped/seg.mgz'),
    # ('CSFnS_invwarped', 'samseg/CSFnS_invwarped/seg.mgz'),
    # ('synth_seg_invwarped', 'synth_seg_invwarped.nii.gz'),
    # ('synth_seg_weighted_invwarped', 'synth_seg_weighted_invwarped.nii.gz'),

    ('WMn', 'samseg/WMn/seg.mgz'),
    
    ('CSFnS', 'samseg/CSFnS/seg.mgz'),
    # ('CSFnSB_wmn', 'samseg/CSFnSB_wmn/seg.mgz'),

    # ('synth_seg', 'synth_seg.nii.gz'),
    ('synth_seg_weighted', 'synth_seg_weighted.nii.gz'),
]

labels = {}
for c, _ in predicted_paths:
    labels[c] = c
labels['synth_seg'] = 'CNN segmentation'
labels['synth_seg_weighted'] = 'CNN segmentation, weighted loss'
labels['CSFnS'] = 'Synthesized CSFn'
labels['CSFnSB_wmn'] = 'Brain-only synthesized CSFn'

data = {}

# predicted_path = 'CSFnSB_wmn_invwarped'

for predicted_path_basename, predicted_path in predicted_paths:
    cache_path = 'dice_cache/{}_vs_{}.pkl'.format(original_path_basename, predicted_path_basename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data[predicted_path_basename] = pickle.load(f)
        continue
    print(predicted_path)

    for subj_path in tqdm(subjs):
        if not os.path.exists(os.path.join(subj_path, predicted_path)):
            continue
        original_seg = nib.load(os.path.join(subj_path, original_path)).get_fdata()
        predicted_seg = nib.load(os.path.join(subj_path, predicted_path)).get_fdata()
        for seg_idx, structure in structures:
            struct_dice_scores[structure].append(dice_score(original_seg, predicted_seg, seg_idx))

    data[predicted_path_basename] = struct_dice_scores
    with open(cache_path, 'wb') as f:
        pickle.dump(struct_dice_scores, f)

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
    # [
    #     '3rd-Ventricle',
    #     '4th-Ventricle',
    #     '5th-Ventricle',
    # ],
]

group_names = [
    'left_deepbrain',
    'right_deepbrain',
    # 'asymmetric_deepbrain',
]

colors = [
    '#2CA02C',
    '#FF7F0E',
    '#448DC0',
]

def subcategorybar(X, vals, stdevs, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    handles = []
    for i in range(n):
        h = plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge", label=labels[predicted_paths[i][0]], yerr=stdevs[i])   
        handles.append(h)
    plt.xticks(_X, X, rotation=45, ha='right')
    return handles

for group_idx, structures in enumerate(structure_groups):
    means = []
    stdevs = []
    for c, _ in predicted_paths:
        submeans = [np.mean(data[c][s]) for s in structures]
        subdevs = [np.std(data[c][s]) for s in structures]
        means.append(submeans)
        stdevs.append(subdevs)

    stdevs=[None] * len(means)

    plt.figure()
    plt.ylim(0, 1)
    handles = subcategorybar(structures, means, stdevs)
    plt.legend(handles=handles, loc='lower left')

    plt.ylabel('Dice score')
    plt.xlabel('Structure')

    plt.tight_layout()

    plt.savefig('graphs/bar_dice_{}.png'.format(group_names[group_idx]))
    plt.clf()


# longest = 'Right-Cerebellum-White-Matter'

# for _, struct in structures:
#     padding = ' ' * (len(longest) - len(struct))
#     print(struct + padding, np.mean(struct_dice_scores[struct]))#, np.std(struct_dice_scores[struct]))
