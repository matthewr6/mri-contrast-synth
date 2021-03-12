import glob
import sys
import os
import re
import shutil

def all_files_in(path):
    if path[-1] == '/':
        path = path[:-1]
    return glob.glob(path + '/*')

def all_subjs_in(path):
    return set([re.search('\d{4}', os.path.basename(f)).group(0) for f in all_files_in(path)])

first_set = all_subjs_in(sys.argv[1])
second_set = all_subjs_in(sys.argv[2])

subjs = first_set.intersection(second_set)

print(len(subjs))

output_path = sys.argv[3]
if output_path[-1] == '/':
    output_path = output_path[:-1]

for subj in subjs:
    subj_path = '{}/{}'.format(output_path, subj)
    os.makedirs(subj_path, exist_ok=True)
    csfn_src = glob.glob(sys.argv[1] + '*' + subj + '*')[0]
    wmn_src = glob.glob(sys.argv[2] + '*' + subj + '*')[0]
    shutil.copy(csfn_src, subj_path + '/CSFn.nii.gz')
    shutil.copy(wmn_src, subj_path + '/WMn.nii.gz')
