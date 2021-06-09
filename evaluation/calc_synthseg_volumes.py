import os
import glob
import numpy as np
import nibabel as nib
import collections
import pickle

from tqdm import tqdm
import numpy as np

paths = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')
vols = [
    'synth_seg.nii.gz',
    'synth_seg_weighted.nii.gz',
]

structures = [
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
    (26, 'Left-Accumbens-area'),
    (28, 'Left-VentralDC'),
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
    (63, 'Right-choroid-plexus'),
    (72, '5th-Ventricle'),
]
structures_dict = dict(structures)

# def invert_corrected_ids(volume):
#     volume_out = np.zeros_like(volume) - 1
#     for idx, (seg_id, _) in enumerate(structures):
#         volume_out[volume == idx] = seg_id
#     return volume_out

data = {}
for vol in vols:
    vol_basename = vol.split('.')[0]
    data[vol_basename] = {}
    for idx, struct in structures_dict.items():
        data[vol_basename][struct] = []

for subj in tqdm(paths):
    original_vol = nib.load(os.path.join(subj, 'samseg/CSFn_registered/seg.mgz')).get_fdata()
    for vol in vols:
        vol_basename = vol.split('.')[0]
        d = nib.load(os.path.join(subj, vol)).get_fdata()
        for idx, struct in structures_dict.items():
            original = (original_vol == idx).sum()
            predicted = (d == idx).sum()
            diff = abs(predicted - original)
            total = predicted + original
            metric_val = 1 - (diff / total)
            data[vol_basename][struct].append(metric_val)

with open('synthseg_volumes_cache.pkl', 'wb') as f:
    pickle.dump(data, f)

