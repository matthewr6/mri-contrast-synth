import os
import glob
import numpy as np
import nibabel as nib

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

# original_path = 'samseg'
# predicted_path = 'samseg_predicted'
original_path = 'samseg_wmn'
predicted_path = 'samseg_warped'

seg_vol_path = 'seg.mgz'

for subj_path in subjs:
    original_seg = nib.load(os.path.join(subj_path, original_path, seg_vol_path)).get_fdata()
    predicted_seg = nib.load(os.path.join(subj_path, predicted_path, seg_vol_path)).get_fdata()
    for seg_idx, structure in structures:
        struct_dice_scores[structure].append(dice_score(original_seg, predicted_seg, seg_idx))

longest = 'Right-Cerebellum-White-Matter'

for _, struct in structures:
    padding = ' ' * (len(longest) - len(struct))
    print(struct + padding, np.mean(struct_dice_scores[struct]))#, np.std(struct_dice_scores[struct]))
