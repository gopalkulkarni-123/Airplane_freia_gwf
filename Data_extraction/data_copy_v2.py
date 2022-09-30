import h5py
import numpy as np
import random as rd
#from mpl_toolkits.mplot3d import plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 1000
dataset_copied = np.zeros((n, 3))

with h5py.File(r'E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\ShapeNet\ShapeNetCore55v2_meshes_resampled_.h5', 'r') as f:
    ls = list(f.keys())
    data = f.get('test_vertices_c')
    datasets = np.array(data)
    
list = np.arange(100, 1000, 1)    

for i in range(n):
    #print(index)
    index = rd.choice(list)
    dataset_copied[i] = datasets[index]
    
#print(dataset_copied)

z = dataset_copied[:,0]
#print(z)
x = dataset_copied[:,1]
#print(x)
y = dataset_copied[:,2]
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