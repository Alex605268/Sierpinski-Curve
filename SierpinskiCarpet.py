import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# just shows how pytorch can be used to make it faster - by doing it iteratively, 
# because each 'pixel' i.e each element in the tensor can be handled independently, the gpu
# can assign a separate thread to each pixel and run them all at the same time, whereas if
# it was the CPU the number of threads is limited


#n controls the size of the square - grid size for sierpinski carpet is always 3^n
n = 1
size = 3 ** n

X = torch.arange(size)
Y = torch.arange(size)

# in documentation this defaults to 'ij' indexing anyway but i'm not sure, np.mgrid defaults to 'xy'
# 
x, y = torch.meshgrid(X, Y, indexing='ij')

sCarpet = torch.ones(size, size)

x = x.to(device)
y = y.to(device)
sCarpet = sCarpet.to(device)


#iteration range =
for i in range(n):
    #this means we do it at each "level" of 3x3 - so remove the middle of every 3*3 square when 
    #n = 1, then the middle of every 9x9 square when n = 2, and so on.
    div = 3 ** i
    #mask is just a matrix of booleans the same size as the carpet. on each iteration we're setting
    #more values to true
    mask = ((x // div) % 3 == 1) & ((y // div) % 3 == 1)
    #this means whenever mask is true, we change that value from 1 to 0, i.e turn that pixel off
    sCarpet[mask] = 0


plt.figure(figsize=(8, 8))
plt.imshow(sCarpet, cmap='binary')
plt.axis('off')
plt.show()

    