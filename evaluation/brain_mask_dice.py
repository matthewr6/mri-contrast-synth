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


def dice_score(seg_a, seg_b):
    X = (seg_a != 0)
    Y = (seg_b != 0)
    X_intersect_Y = X & Y
    return 2 * X_intersect_Y.sum() / (X.sum() + Y.sum())

subjs = glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*')

original_path = 'CSFnB.nii.gz'
predicted_path = 'CSFnSB_wmn_invwarped.nii.gz'
# predicted_path = 'WMnB_direct_invwarped.nii.gz'

brainmask_dice = []
cache_path = 'dice_cache/brainmask_{}_vs_{}.pkl'.format(original_path, predicted_path)
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        brainmask_dice = pickle.load(f)
else:
    for subj_path in tqdm(subjs):
        if not os.path.exists(os.path.join(subj_path, predicted_path)):
            continue
        original_seg = nib.load(os.path.join(subj_path, original_path)).get_fdata()
        predicted_seg = nib.load(os.path.join(subj_path, predicted_path)).get_fdata()
        brainmask_dice.append(dice_score(original_seg, predicted_seg))

with open(cache_path, 'wb') as f:
    pickle.dump(brainmask_dice, f)

print(np.mean(brainmask_dice))
