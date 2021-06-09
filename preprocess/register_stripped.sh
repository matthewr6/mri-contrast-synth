#!/bin/bash

antsRegistration -v -d 3 --float 0 --output stripped_wmn_output --use-histogram-matching 1 -t Rigid[0.1] --metric Mattes[WMnB.nii.gz,CSFnB.nii.gz,1,32,None] --convergence [500x500x500x500x500,4e-7,10] -f 5x5x5x5x4 -s 1.685x1.4771x1.256x1.0402x0.82235mm -t Affine[0.1] --metric Mattes[WMnB.nii.gz,CSFnB.nii.gz,1,64, None] --convergence [450x150x50,1e-7,10] -f 3x2x1 -s 0.60056x0.3677x0mm -t SyN[0.4,3.0] --metric MI[WMnB.nii.gz,CSFnB.nii.gz,1,32,None] --convergence [200x200x90x50,1e-10,10] -f 4x3x2x1 -s 0.82x0.6x0.3677x0.0mm

WarpImageMultiTransform 3 CSFnB.nii.gz CSFnB_registered_wmn.nii.gz -R WMnB.nii.gz stripped_wmn_output1Warp.nii.gz stripped_wmn_output0GenericAffine.mat
