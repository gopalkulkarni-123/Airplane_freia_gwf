import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

with h5py.File('ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    ls = list(f.keys())
    data = f.get('test_vertices_c')
    datasets = np.array(data)
    #print(f.keys.test_vertices_c)

