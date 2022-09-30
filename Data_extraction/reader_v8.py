import h5py
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

with h5py.File('ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    print(f.keys())
    #ls = list(f.keys())
    #data = f.get('test_vertices_c')
    #datasets = np.array(data)
    #print(f.keys.test_vertices_c)

"""
z = datasets[:,0]
#print(z)
x = datasets[:,1]
#print(x)
y = datasets[:,2]
#print(y)

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(x, y, z, color = "green")
plt.title("simple 3D scatter plot")
 
# show plot
plt.show()


with open('airplane_data_v4.txt', 'w') as k:

    for i  in range (10000):
       k.write(str(datasets[i])) 
       
"""
#print(datasets[1])
       