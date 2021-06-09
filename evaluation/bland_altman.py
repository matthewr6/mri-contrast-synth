import os
import glob
from tqdm import tqdm
import numpy as np
import nibabel as nib
import collections

import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.patches as mpatches

structures = [
    'Unknown',
    'Left-Cerebral-White-Matter',
    'Left-Cerebral-Cortex',
    'Left-Lateral-Ventricle',
    'Left-Inf-Lat-Vent',
    'Left-Cerebellum-White-Matter',
    'Left-Cerebellum-Cortex',
    'Left-Thalamus',
    'Left-Caudate',
    'Left-Putamen',
    'Left-Pallidum',
    '3rd-Ventricle',
    '4th-Ventricle',
    'Brain-Stem',
    'Left-Hippocampus',
    'Left-Amygdala',
    'CSF',
    'Left-Accumbens-area',
    'Left-VentralDC',
    'Left-vessel',
    'Left-choroid-plexus',
    'Right-Cerebral-White-Matter',
    'Right-Cerebral-Cortex',
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
    'Right-vessel',
    'Right-choroid-plexus',
    '5th-Ventricle',
    'WM-hypointensities',
    'non-WM-hypointensities',
    'Optic-Chiasm',
    'Skull',
    'Head-ExtraCerebral',
    'Eye-Fluid',
]

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

subjs = list(glob.glob('/data/mradovan/7T_WMn_3T_CSFn_pairs/*'))
subj_ids = [s.split('/')[-1] for s in subjs]

outliers = ['0733', '0762', '1196']
subj_ids = [s for s in subj_ids if s not in outliers]

data_paths = [
    'samseg',
    'CSFn_out_single_slice_invwarped',
    'CSFn_out_single_slice_vgg_invwarped',
    'CSFn_out_single_slice_perceptual_invwarped',
]
ground_truth = 'samseg'

data = {}

for path in data_paths:
    volumes = collections.defaultdict(list)

    for subj_path in subjs:
        vols = samseg_stats_to_dict(os.path.join(subj_path, path))
        for structure, volume in vols.items():
            volumes[structure].append(volume)

    data[path] = volumes

structures = data[data_paths[0]].keys()

def bland_altman_plot(data1, data2, labels, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(
        tkr.FuncFormatter(lambda y,  p: format(int(y), ',')))
    ax.xaxis.set_major_formatter(
        tkr.FuncFormatter(lambda x,  p: format(int(x), ',')))
    
    for idx, label in enumerate(labels):
        plt.annotate(label, (mean[idx], diff[idx]))

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

for structure in tqdm(structures):
    plt.clf()
    bland_altman_plot(data['samseg'][structure], data['CSFn_out_single_slice_invwarped'][structure], subj_ids)
    plt.title(structure)
    plt.savefig('graphs/blandaltman/{}.png'.format(structure))
