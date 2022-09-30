import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

with h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    
    print(f.keys())
    #ls = list(f.keys())
    
    
    """
    data = f.get('train_labels')
    datasets = np.array(data)
    for i in range(datasets.size):
        print(datasets[i])
    """
    #print(f.keys.test_vertices_c)
    #print(f.get('test_labels'))