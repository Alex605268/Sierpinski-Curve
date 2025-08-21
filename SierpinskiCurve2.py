import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#this generates a sierpinski curve 

def s_curve(levels, path):
    # base case
    if levels == 0:
        return path
    
    #extract the 3 vertices of the triangle
    v1 = path[0]
    v2 = path[1]
    v3 = path[2]
    
    #each shift scales down the triangle by 50%, anchored at a different vertex
    #so we end up with 3 new triangular paths, half the size, each anchored at a different vertex
    shifts  = [
        lambda z: v1 + (z-v1) * 0.5,
        lambda z: v2 + (z-v2) * 0.5,
        lambda z: v3 + (z-v3) * 0.5
    ]
    
    #recursive case
    sub_paths = []
    for shift in shifts:
        #essentially this just keeps doing the shift operations on smaller and smaller triangles
        #until it reaches the base case (where levels == 0) then appends them all together.
        sub_path = s_curve(levels-1, shift(path))
        print(f"Appending at level {levels}")
        sub_paths.append(sub_path)
        
        #ai gave me this part - it was attaching a few extra lines that I
        #didn't want it to, and couldn't work how to fix it. this adds a "stop" at the end of each
        #small triangle so that matplotlib doesn't try and attach it to a bigger triangle
        sub_paths.append(torch.tensor([float('nan') + float('nan')*1j]))
        
    return torch.cat(sub_paths)

#the three point of the triangle path (currently it's an equilateral triangle)
#function should be able to handle any points.
point0 = torch.tensor([0.0, 0.0])
point1 = torch.tensor([1.0, 0.0])
point2 = torch.tensor([0.5, 0.866])

#creates the triangle path (stack puts those 3 tensors on top of each other in a new dimension, so
# 3 1*2 matrices becomes a 3*2 matrix)
path = torch.stack([point0, point1, point2, point0])

# converting the points to the complex plane
z = torch.complex(path[:, 0], path[:, 1])
z = z.to(device)

myPath = z
lvl = 3

curve = s_curve(lvl, myPath)


plt.plot(curve.real.cpu().numpy(), curve.imag.cpu().numpy(), color='blue', linewidth=0.5)
plt.axis('equal')
plt.axis('off')
plt.show()
