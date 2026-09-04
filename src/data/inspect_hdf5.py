import h5py

file_path = "all_patches.hdf5"

with h5py.File(file_path, "r") as f:
    print("Keys in HDF5 file:")
    print(list(f.keys()))
    
    # Print dataset shapes
    for key in f.keys():
        print(f"\nDataset: {key}")
        print("Shape:", f[key].shape)
import numpy as np

with h5py.File(file_path, "r") as f:
    labels = f["slice_class"][:]
    unique_labels = np.unique(labels)
    print("\nUnique labels:", unique_labels)
