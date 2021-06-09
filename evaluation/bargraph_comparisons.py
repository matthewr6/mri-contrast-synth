import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def parse_file(fname, structures):
    with open(fname, 'r') as f:
        data = f.readlines()
    data = [d.strip().split(' ') for d in data]
    data = [list(filter(None, d)) for d in data]
    data_dict = {}
    for d in data:
        struct = d[0]
        vol_diff = d[1]
        data_dict[struct] = vol_diff
    ret = []
    for struct in structures:
        ret.append(float(data_dict[struct]))
    return ret

compares = [
    'CSFnS_invwarped',
    'CSFnSB_wmn_invwarped',
]

for group_idx, structures in enumerate(structure_groups):
    data = [parse_file(os.path.join(c, 'dice.txt'), structures) for c in compares]

    def subcategorybar(X, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        handles = []
        for i in range(n):
            h = plt.barh(_X - width/2. + i/float(n)*width, vals[i], 
                    height=width/float(n), align="edge", label=compares[i])   
            handles.append(h)
        plt.yticks(_X, X)
        return handles

    plt.figure()
    plt.xlim(0, 1)
    handles = subcategorybar(structures, data)
    plt.legend(handles=handles[::-1])

    plt.ylabel('Dice score')
    plt.xlabel('Structure')

    plt.tight_layout()

    plt.savefig('graphs/bar_dice_{}.png'.format(group_names[group_idx]))
    plt.clf()