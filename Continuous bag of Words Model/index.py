import numpy as np

## ACTIVATION FUNCTIONS
# ReLU
np.random.seed(10)

z_1 = 10 * np.random.rand(5,1)-5 #random 5x1 col. vector\
print(z_1)

h = z_1.copy()
h[h<0]=0
print(h)
def ReLU(z):
    r = z.copy()
    r[r<0] = 0
    return r
z = np.array([[3.1415],[2.9107], [-1.4356],[0.2349]])
print(ReLU(z))

print("\n\n\n")

# SoftMax
def SoftMAX(z):
    e = np.exp(z)
    sum_e = np.sum(e)
    return e/sum_e
print(SoftMAX([9, 8, 11, 10, 8.5]))
print(np.sum(SoftMAX([9, 8, 11, 10, 8.5])) ==1)


## 1-D ARRYS VS 2-D Column VECTORS
V = 4
xArray = np.zeros(V)
xColVector = xArray.copy()
xColVector.shape = (V,1)
print(xColVector)
