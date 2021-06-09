import numpy as np
import nibabel as nib
import glob
import random
import sys
import os
import re
from tqdm import tqdm
import itertools
import math

slices_per_slab = 5
slices_per_volume = 440

input_modes = {
    'full': ('{}/WMn.nii.gz', '{}/samseg/CSFn_registered/seg.mgz'),
    'stripped_wmn': ('{}/WMnB.nii.gz', '{}/samseg/CSFnB_wmn_registered/seg.mgz'),
}

wmn_format, seg_format = input_modes['full']

def set_formats(input_mode):
    global wmn_format
    global seg_format
    wmn_format, seg_format = input_modes[input_mode]

# 12 structures: left brain
# structures = [
#     # (0, 'Unknown'),
#     (2, 'Left-Cerebral-White-Matter'),
#     (3, 'Left-Cerebral-Cortex'),
#     # (4, 'Left-Lateral-Ventricle'),
#     # (5, 'Left-Inf-Lat-Vent'),
#     (7, 'Left-Cerebellum-White-Matter'),
#     (8, 'Left-Cerebellum-Cortex'),
#     (10, 'Left-Thalamus'),
#     (11, 'Left-Caudate'),
#     (12, 'Left-Putamen'),
#     (13, 'Left-Pallidum'),
#     # (14, '3rd-Ventricle'),
#     # (15, '4th-Ventricle'),
#     # (16, 'Brain-Stem'),
#     (17, 'Left-Hippocampus'),
#     (18, 'Left-Amygdala'),
#     # (24, 'CSF'),
#     (26, 'Left-Accumbens-area'),
#     # (28, 'Left-VentralDC'),
#     # (30, 'Left-vessel'),
#     (31, 'Left-choroid-plexus'),
#     # (41, 'Right-Cerebral-White-Matter'),
#     # (42, 'Right-Cerebral-Cortex'),
#     # (43, 'Right-Lateral-Ventricle'),
#     # (44, 'Right-Inf-Lat-Vent'),
#     # (46, 'Right-Cerebellum-White-Matter'),
#     # (47, 'Right-Cerebellum-Cortex'),
#     # (49, 'Right-Thalamus'),
#     # (50, 'Right-Caudate'),
#     # (51, 'Right-Putamen'),
#     # (52, 'Right-Pallidum'),
#     # (53, 'Right-Hippocampus'),
#     # (54, 'Right-Amygdala'),
#     # (58, 'Right-Accumbens-area'),
#     # (60, 'Right-VentralDC'),
#     # (62, 'Right-vessel'),
#     # (63, 'Right-choroid-plexus'),
#     # (72, '5th-Ventricle'),
#     # (77, 'WM-hypointensities'),
#     # (80, 'non-WM-hypointensities'),
#     # (85, 'Optic-Chiasm'),
#     # (165, 'Skull'),
#     # (258, 'Head-ExtraCerebral'),
#     # (259, 'Eye-Fluid'),
# ]
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
ignored_structures = [
    (0, 'Unknown'),
    (24, 'CSF'),
    (30, 'Left-vessel'),
    (62, 'Right-vessel'),
    (77, 'WM-hypointensities'),
    (80, 'non-WM-hypointensities'),
    (85, 'Optic-Chiasm'),
    (165, 'Skull'),
    (258, 'Head-ExtraCerebral'),
    (259, 'Eye-Fluid'),
]
n_structures = len(structures)

def correct_seg_ids(volume):
    volume_out = np.zeros_like(volume) + n_structures
    for idx, (seg_id, _) in enumerate(structures):
        volume_out[volume == seg_id] = idx
    return volume_out

def invert_corrected_ids(volume):
    volume_out = np.zeros_like(volume) - 1
    for idx, (seg_id, _) in enumerate(structures):
        volume_out[volume == idx] = seg_id
    return volume_out

def find_identifiers(path):
    files = [p for p in glob.glob(path + '/*')]
    identifiers = set()
    for f in files:
        identifier = re.sub('[^0-9]', '', os.path.basename(f))
        if identifier and os.path.exists(wmn_format.format(f)) and os.path.exists(seg_format.format(f)):
            identifiers.add(identifier)
    return list(sorted(identifiers))

nifti_cache = {}
def load_nifti(path, _normalize=False, _correct_seg_ids=False):
    if path not in nifti_cache:
        nifti_cache[path] = nib.load(path).get_fdata()
        if _normalize:
            nifti_cache[path] = normalize(nifti_cache[path])
        elif _correct_seg_ids:
            nifti_cache[path] = correct_seg_ids(nifti_cache[path])
    return nifti_cache[path]

def preload_volumes(identifiers, build_vol_paths, get_opts=None, vols_per_id=None):
    if vols_per_id:
        print('Preloading {} volumes...'.format(vols_per_id * len(identifiers)))
    else:
        print('Preloading {} identifiers...')
    paths = [p for i in identifiers for p in build_vol_paths(i)]
    for path in tqdm(paths):
        if get_opts:
            load_nifti(path, **get_opts(path))
        else:
            load_nifti(path)
    print('Preloading completed')

def normalize(x, percentile=95):
    robust_max = np.percentile(x, percentile)
    # x[x > robust_max] = robust_max # Chris said he doesn't clip
    # return (x - x.min()) / (x.max() - x.min())
    return (x - x.min()) / (robust_max - x.min())

def get_shape(base_path, identifier):
    path = base_path + '/' + wmn_format.format(identifier)
    if path not in nifti_cache:
        path = base_path + '/' + seg_format.format(identifier)
    return nifti_cache[path].shape

def calculate_weights(identifiers, build_vol_paths):
    paths = [p for i in identifiers for p in build_vol_paths(i)]
    paths = [p for p in paths if 'seg.mgz' in p]
    struct_proportions = {}
    print('Calculating structure weights...')
    for p in paths:
        vol = load_nifti(p, _correct_seg_ids=True)
        vol_total = np.prod(vol.shape)
        for struct_idx in range(n_structures + 1):
            struct_total = (vol == struct_idx).sum()
            if struct_idx not in struct_proportions:
                struct_proportions[struct_idx] = []

            metric = 1 - (struct_total / vol_total) # want smaller structures to have higher weight

            struct_proportions[struct_idx].append(metric)
    for struct_idx in struct_proportions:
        struct_proportions[struct_idx] = np.mean(struct_proportions[struct_idx])
    return struct_proportions

def seg_generator(base_path, batch_size=1, n_volumes=10):
    identifiers = find_identifiers(base_path)[:n_volumes]
    def build_vol_paths(identifier):
        return [
            base_path + '/' + wmn_format.format(identifier),
            base_path + '/' + seg_format.format(identifier)
        ]
    def get_opts(path):
        if 'seg.mgz' in path:
            return { '_correct_seg_ids': True }
        if '.nii.gz' in path:
            return { '_normalize': True }
        return {}
    preload_volumes(identifiers, build_vol_paths, get_opts=get_opts, vols_per_id=2)
    weights = calculate_weights(identifiers, build_vol_paths)
    identifier_slice_pairs = [(identifier, slice_idx) for identifier in identifiers for slice_idx in range(get_shape(base_path, identifier)[1] - 4)]
    random.shuffle(identifier_slice_pairs)
    N = len(identifier_slice_pairs)
    batches_per_epoch = math.ceil(N / batch_size)

    def generator():
        i = 0
        while i < N:
            X = []
            y = []
            next_batch_size = min(batch_size, N - i)
            for _ in range(next_batch_size):
                identifier = identifier_slice_pairs[i][0]
                slice_idx = identifier_slice_pairs[i][1]

                wmn_path = base_path + '/' + wmn_format.format(identifier)
                seg_path = base_path + '/' + seg_format.format(identifier)

                wmn = load_nifti(wmn_path, _normalize=True)
                seg = load_nifti(seg_path, _correct_seg_ids=True)

                wmn_slab = np.zeros((256, 5, 256))
                seg_slab = np.zeros((256, 5, 256))

                wmn_slab = wmn[:, slice_idx:slice_idx + slices_per_slab, :]
                seg_slab = seg[:, slice_idx:slice_idx + slices_per_slab, :]

                wmn_slab = np.rollaxis(wmn_slab, 2)
                seg_slab = np.rollaxis(seg_slab, 2)

                X_ = None
                y_ = None

                X_ = wmn_slab[:, :, :, None]
                y_ = seg_slab[:, :, 2]
                y_ = y_.reshape((y_.shape[0] * y_.shape[1]))

                X.append(X_)
                y.append(y_)
                    
                i += 1
                
            yield np.array(X), np.array(y)

    return generator, batches_per_epoch, weights

if __name__ == '__main__':
    path = '/data/mradovan/7T_WMn_3T_CSFn_pairs'
    generator, _ = seg_generator(path, batch_size=10, n_volumes=1)
    g = generator()
    for i in range(100):
        X, y = next(g)
        # print(X[0].shape, y[0].shape)
        print(np.unique(y[0]))
