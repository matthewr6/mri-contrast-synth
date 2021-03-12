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

wmn_format = '{}/WMn.nii.gz'
csfn_format = '{}/CSFn_registered.nii.gz'
# wmn_format = '{}/WMn_stripped.nii.gz'
# csfn_format = '{}/CSFn_stripped_registered.nii.gz'

def find_identifiers(path):
    files = [p for p in glob.glob(path + '/*')]
    identifiers = set()
    for f in files:
        identifier = re.sub('[^0-9]', '', os.path.basename(f))
        if identifier and os.path.exists(wmn_format.format(f)) and os.path.exists(csfn_format.format(f)):
            identifiers.add(identifier)
    return list(sorted(identifiers))

nifti_cache = {}
def load_nifti(path):
    if path not in nifti_cache:
        nifti_cache[path] = normalize(nib.load(path).get_fdata())
    return nifti_cache[path]

def preload_volumes(identifiers, build_vol_paths, vols_per_id=None):
    if vols_per_id:
        print('Preloading {} volumes...'.format(vols_per_id * len(identifiers)))
    else:
        print('Preloading {} identifiers...')
    paths = [p for i in identifiers for p in build_vol_paths(i)]
    for path in tqdm(paths):
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
        path = base_path + '/' + csfn_format.format(identifier)
    return nifti_cache[path].shape

def corrupt_by_scramble(data, corruption_size=(12, 12)):
    patch_x = np.random.randint(data.shape[0] - corruption_size[0])
    patch_y = np.random.randint(data.shape[1] - corruption_size[1])
    patch = data[patch_x:patch_x + corruption_size[0], patch_y:patch_y + corruption_size[1]].flatten()
    np.random.shuffle(patch)
    patch = np.reshape(patch, corruption_size)
    corrupted = data.copy()
    corrupted[patch_x:patch_x + corruption_size[0], patch_y:patch_y + corruption_size[1]] = patch
    return corrupted


def eval_generator(base_path, mode, corrupted=False, batch_size=1, n_volumes=10):
    assert mode in ['csfn', 'wmn']
    
    identifiers = find_identifiers(base_path)[-n_volumes:]
    def build_vol_paths(identifier):
        if mode == 'csfn':
            path = base_path + '/' + csfn_format.format(identifier)
        else:
            path = base_path + '/' + wmn_format.format(identifier)
        return [path]
    preload_volumes(identifiers, build_vol_paths, vols_per_id=1)
    identifier_slice_pairs = [(identifier, slice_idx) for identifier in identifiers for slice_idx in range(get_shape(base_path, identifier)[1])]
    random.shuffle(identifier_slice_pairs)
    N = len(identifier_slice_pairs)
    batches_per_epoch = math.ceil(N / batch_size)

    def generator():
        i = 0
        while i < N:
            X = []
            next_batch_size = min(batch_size, N - i)
            for _ in range(next_batch_size):
                identifier = identifier_slice_pairs[i][0]
                slice_idx = identifier_slice_pairs[i][1]

                vol_path = base_path + '/'
                if mode == 'csfn':
                    vol_path += csfn_format.format(identifier)
                elif mode == 'wmn':
                    vol_path += wmn_format.format(identifier)
                vol = nifti_cache[vol_path]

                vol_slice = vol[:, slice_idx, :]

                # to keep consistent with np.transpose(slice) in corrupted_with_wmn_generator
                vol_slice = np.transpose(vol_slice)

                if corrupted:
                    vol_slice = corrupt_by_scramble(vol_slice)

                X.append(vol_slice[:, :, None])
                i += 1

            yield np.array(X)

    return generator, batches_per_epoch


def corrupted_with_wmn_generator(base_path, batch_size=1, n_volumes=10):
    identifiers = find_identifiers(base_path)[:n_volumes]
    def build_vol_paths(identifier):
        return [
            base_path + '/' + wmn_format.format(identifier),
            base_path + '/' + csfn_format.format(identifier)
        ]
    preload_volumes(identifiers, build_vol_paths, vols_per_id=2)
    identifier_slice_pairs = [(identifier, slice_idx) for identifier in identifiers for slice_idx in range(get_shape(base_path, identifier)[1])]
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
                csfn_path = base_path + '/' + csfn_format.format(identifier)

                wmn = nifti_cache[wmn_path]
                csfn = nifti_cache[csfn_path]

                wmn_slice = np.zeros((256, 1, 256))
                csfn_slice = np.zeros((256, 1, 256))

                wmn_slice = wmn[:, slice_idx, :]
                csfn_slice = csfn[:, slice_idx, :]

                # to keep consistent with np.rollaxis(slab, 2) in paired_generator
                wmn_slice = np.transpose(wmn_slice)
                csfn_slice = np.transpose(csfn_slice)

                X.append(csfn_slice[:, :, None])
                y.append(1)
                X.append(corrupt_by_scramble(csfn_slice)[:, :, None])
                y.append(0)
                X.append(wmn_slice[:, :, None])
                y.append(0)

                i += 1
                
            yield np.array(X), np.array(y)
    return generator, batches_per_epoch

def paired_generator(base_path, batch_size=1, n_volumes=10):
    identifiers = find_identifiers(base_path)[:n_volumes]
    def build_vol_paths(identifier):
        return [
            base_path + '/' + wmn_format.format(identifier),
            base_path + '/' + csfn_format.format(identifier)
        ]
    preload_volumes(identifiers, build_vol_paths, vols_per_id=2)
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
                csfn_path = base_path + '/' + csfn_format.format(identifier)

                wmn = load_nifti(wmn_path)
                csfn = load_nifti(csfn_path)

                wmn_slab = np.zeros((256, 5, 256))
                csfn_slab = np.zeros((256, 5, 256))

                wmn_slab = wmn[:, slice_idx:slice_idx + slices_per_slab, :]
                csfn_slab = csfn[:, slice_idx:slice_idx + slices_per_slab, :]

                wmn_slab = np.rollaxis(wmn_slab, 2)
                csfn_slab = np.rollaxis(csfn_slab, 2)

                X.append(wmn_slab[:, :, :, None])
                y.append(csfn_slab[:, :, 2, None])

                i += 1
                
            yield np.array(X), np.array(y)

    return generator, batches_per_epoch

if __name__ == '__main__':
    path = '/data/mradovan/7T_WMn_3T_CSFn_pairs'
    generator = paired_generator(path, batch_size=10)
    X, y_img, y_features = next(generator)
    print(X[0].shape, y_img[0].shape, y_features[0].shape)
