import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

vol = nib.load(sys.argv[1]).get_fdata()

vmin = vol.min()
vmax = vol.max()

print(vmin, vmax)

plt.figure()
vol_slice = np.rot90(vol[int(vol.shape[0] / 2) + 10, :, :])
plt.imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
# plt.show()

plt.figure()
vol_slice = np.rot90(vol[:, int(vol.shape[1] / 2) + 10, :])
plt.imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
# plt.show()

plt.figure()
vol_slice = np.rot90(vol[:, :, int(vol.shape[2] / 2) + 10])
plt.imshow(vol_slice, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()

