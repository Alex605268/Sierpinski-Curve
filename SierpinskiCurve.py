import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#

def s_curve(levels, shape):
    
    if levels == 0:
        return torch.stack(shape + [shape[0]])
    
    #finds the midpoints of each line in the shape
    mid01 = (shape[0] + shape[1]) / 2
    mid12 = (shape[1] + shape[2]) / 2
    mid20 = (shape[2] + shape[0]) / 2
    
    #creats 3 smaller shapes based on the midpoints of the bigger shape
    newShape1 = [shape[0], mid01, mid20]
    newShape2 = [mid01, shape[1], mid12]
    newShape3 = [mid20, mid12, shape[2]]
    
    #recursively generate smaller s_curves
    subCurve1 = s_curve(levels - 1, newShape1)
    subCurve2 = s_curve(levels - 1, newShape2)
    subCurve3 = s_curve(levels - 1, newShape3)
    
    
    #ai provided this line - I couldn't work out why random extra lines were being generated
    #it turns out that the final point of the final smaller triangle was being concatenated to
    #the starting point of the first larger triangles because of the recursive call.This resulted
    # in some lines being doubly drawn, as well as a few extra lines that shouldn't appear
    # matplotlib considers NaN a discontinuity so by adding it between each triangle it prevents the extra lines being drawn.
    nan = torch.tensor([[float('nan'), float('nan')]])
    newShape = [subCurve1, nan, subCurve2, nan, subCurve3]
    
    # newShape = [subCurve1, subCurve2, subCurve3]
    
    return torch.cat(newShape, dim=0)


point0 = torch.tensor([0.0, 0.0])
point1 = torch.tensor([1.0, 0.0])
point2 = torch.tensor([0.5, 0.866])

point0 = point0.to(device)
point1 = point1.to(device)
point2 = point2.to(device)

myShape = [point0, point1, point2]
lvl = 8

curve = s_curve(lvl, myShape)

plt.plot(curve[:, 0], curve[:, 1], color='blue', linewidth=0.5)
plt.axis('equal')
plt.axis('off')
plt.show()
