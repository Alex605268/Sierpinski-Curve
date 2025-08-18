import torch
import numpy as np
import matplotlib.pyplot as plt


def s_curve(levels, shape):
    
    if levels == 0:
        return torch.stack(shape + [shape[0]])
    
    mid01 = (shape[0] + shape[1]) / 2
    mid12 = (shape[1] + shape[2]) / 2
    mid20 = (shape[2] + shape[0]) / 2
    
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

myShape = [point0, point1, point2]
lvl = 2

curve = s_curve(lvl, myShape)

torch.set_printoptions(threshold=10000)
print(curve)

plt.plot(curve[:, 0], curve[:, 1], color='blue', linewidth=0.5)
plt.axis('equal')
plt.axis('off')
plt.title("Sierpinski Curve (PyTorch)")
plt.show()
    
        
        
    
        
        




# fig = plt.figure(figsize=(16,10))
# def processFractal(a):
#     """Display an array of iteration counts as a
#         colorful picture of a fractal."""
#     a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
#     img = np.concatenate([10+20*np.cos(a_cyclic),
#     30+50*np.sin(a_cyclic),
#     155-80*np.cos(a_cyclic)], 2)
#     img[a==a.max()] = 0
#     a = img
#     a = np.uint8(np.clip(a, 0, 255))
#     return a

# plt.imshow(processFractal(ns.cpu().numpy()))
# plt.tight_layout(pad=0)
# plt.show()