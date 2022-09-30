import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

n = 100000
dataset_copied = np.zeros((n, 3))

with h5py.File('ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    ls = list(f.keys())
    data = f.get('test_vertices_c')
    datasets = np.array(data)
    
for i in range(100):
    dataset_copied[i] = datasets[i]
    
#print(dataset_copied)

z = dataset_copied[:,0]
#print(z)
x = dataset_copied[:,1]
#print(x)
y = dataset_copied[:,2]
#print(y)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()   