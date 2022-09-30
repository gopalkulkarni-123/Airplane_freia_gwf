import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r') as fin:
    chosen_label = 0
    chosen_label_ind = (np.array(fin['train_labels'],dtype = np.uint8)==chosen_label).nonzero()[0]
    vertices_c_bounds = np.empty(fin['train_vertices_c_bounds'].shape, dtype=np.uint64)
    fin['train_vertices_c_bounds'].read_direct(vertices_c_bounds)
    #print("chosen_label_ind", chosen_label_ind)
    print("vertices_c_bounds", vertices_c_bounds)
    
    
#i = chosen_label_ind[1]
#print(chosen_label_ind[1])

data_file = h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r')
vertices_c = np.array(data_file['train_vertices_c'][vertices_c_bounds[0]:vertices_c_bounds[1]], dtype=np.float32)
#a = np.array[vertices_c_bounds[0]:vertices_c_bounds[1]]
#print(vertices_c_bounds[0])
print(vertices_c)

#Plotting
z = vertices_c[:,0]
#print(z)
x = vertices_c[:,1]
#print(x)
y = vertices_c[:,2]
#print(y)
"""

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()   
"""
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z)

plt.show()