import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r') as fin:
    chosen_label = 0
    chosen_label_ind = (np.array(fin['train_labels'],dtype = np.uint8)==chosen_label).nonzero()[0]
    
    vertices_c_bounds = np.empty(fin['train_vertices_c_bounds'].shape, dtype=np.uint64)
    fin['train_vertices_c_bounds'].read_direct(vertices_c_bounds)
    
with open(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\airplane_data_v9.txt', 'w') as k:
    for m in range(0,9):
        print(m)
        data_file = h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r')
        vertices_c = np.array(data_file['train_vertices_c'][vertices_c_bounds[m]:vertices_c_bounds[m+1]], dtype=np.float32)

        k_ = int(vertices_c.size/3)
        iter = k_

     
        if m == 0:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 1:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 2:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 3:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 4:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 5:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 6:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")
        if m == 7:
                for i in range (iter):
                    k.write(str(vertices_c[i][0])+", "+str(vertices_c[i][1])+", "+str(vertices_c[i][2])+","+ str(m) +"\n")