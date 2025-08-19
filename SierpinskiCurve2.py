import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def s_curve(levels, shape):
    
    if levels == 0:
        return torch.cat([shape, shape[0:1]])
    
    #designates the midpoint of each line
    mid01 = (shape[0] + shape[1]) / 2
    mid12 = (shape[1] + shape[2]) / 2
    mid20 = (shape[2] + shape[0]) / 2
    
    
    #creates 3 sub-triangles
    newShape1 = torch.stack([shape[0], mid01, mid20])
    newShape2 = torch.stack([mid01, shape[1], mid12])
    newShape3 = torch.stack([mid20, mid12, shape[2]])
    
    #recursively generate smaller s_curves
    subCurve1 = s_curve(levels - 1, newShape1)
    subCurve2 = s_curve(levels - 1, newShape2)
    subCurve3 = s_curve(levels - 1, newShape3)
    
    
    # ai provided this line - I couldn't work out why random extra lines were being generated
    # it turns out that the final point of the final smaller triangle was being concatenated to
    # the starting point of the first larger triangles because of the recursive call.This resulted
    # in some lines being doubly drawn, as well as a few extra lines that shouldn't appear
    # matplotlib considers NaN a discontinuity so by adding it between each triangle it 
    # prevents the extra lines being drawn.
    nan = torch.tensor([float('nan') + float('nan')*1j])
    newShape = [subCurve1, nan, subCurve2, nan, subCurve3]
    return torch.cat(newShape, dim=0)

#the three outside points of the triangle - can pick any coordinates you want. current coordinates
#are for an equilateral triangle with side length 1
point0 = [0.0, 0.0]
point1 = [1.0, 0.0]
point2 = [0.5, 0.866]

#converting the points to the complex plane
x = torch.tensor([point0[0], point1[0], point2[0]])
y = torch.tensor([point0[1], point1[1], point2[1]])
z = torch.complex(x, y)

z = z.to(device)

myShape = z
lvl = 7

curve = s_curve(lvl, myShape)


#mobius transfomation function
#I just googled interesting complex transformations and found the formula for mobius transformations
#can't have ad-bc = 0
#can stretch/bend/invert the triangle but preserves the angles
#Not entirely sure what each variable does, just found it interesting
def mobius(z, a, b, c, d):
    return (a*z + b) / (c*z + d)

a = 1 + 0.5j
b = -0.3 + 0.2j
c = 0.6 - 0.4j
d = 1 - 0.1j


transformed = mobius(curve, a, b, c, d) 


# for plotting a single curve

# plt.plot(curve.real.cpu(), curve.imag.cpu(), color='blue', linewidth=0.5)
# plt.axis('equal')
# plt.axis('off')
# plt.show()

# optional for standard and transformation side by side

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(curve.real.cpu(), curve.imag.cpu(), color='blue', linewidth=0.5)
ax2.plot(transformed.imag.cpu(), transformed.real.cpu(), color='blue', linewidth=0.5)
plt.tight_layout()
plt.show()

