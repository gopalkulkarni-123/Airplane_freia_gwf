import h5py
import numpy as np
with h5py.File('ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    ls = list(f.keys())
    data = f.get('test_vertices_c')
    datasets = np.array(data)
    #print(f.keys.test_vertices_c)
    
with open('airplane_reader_v1.txt', 'w') as k:
   for i in range(datasets.shape[0]):
        #print(datasets[i])
        k.write(str(datasets[i]))
    