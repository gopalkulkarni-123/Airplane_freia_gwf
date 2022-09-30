import random as rd
import numpy as np

n = 10000
list = np.arange(0, n, 1)
index = rd.choice(list)
print(index)