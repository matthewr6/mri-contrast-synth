import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

vol = nib.load(sys.argv[1]).get_fdata()
vmax = np.max(vol)

print(np.unique(vol))

center = (int(vol.shape[0] / 2) + 10, int(vol.shape[1] / 2) + 10, int(vol.shape[2] / 2) + 10)

plt.figure()
vol_slice = np.rot90(vol[center[0], :, :])
plt.imshow(vol_slice, cmap='tab20b', vmin=0, vmax=vmax)
# plt.show()

plt.figure()
vol_slice = np.rot90(vol[:, center[1], :])
plt.imshow(vol_slice, cmap='tab20b', vmin=0, vmax=vmax)
# plt.show()

plt.figure()
vol_slice = np.rot90(vol[:, :, center[2]])
plt.imshow(vol_slice, cmap='tab20b', vmin=0, vmax=vmax)
plt.show()
