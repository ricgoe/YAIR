import pickle
import matplotlib.pyplot as plt
import numpy as np

path = 'ds_32_32_INTER_AREA/batch_0000.pkl'
path1 = 'ds_32_32/batch_0000.pkl'

with open(path, 'rb') as file:
    arr = pickle.load(file)
with open(path1, 'rb') as file:
    arr1 = pickle.load(file)

index = 12859
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(6,6))
    
#print(np.min(arr[:, 0]), np.max(arr[:,0]))
ax1.imshow(arr[index,0], cmap='grey')
ax1.set_title('Inter Area')
ax2.imshow(arr1[index,0], cmap='grey')
ax2.set_title('Cubic')
plt.show()