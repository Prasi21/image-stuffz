from cv2 import waitKey, warpPerspective
from skimage import data, transform
import numpy as np  
import matplotlib.pyplot as plt
import cv2
import math

def matrixMultiply(X, Y):
    result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    return result


A = np.array([[1,2],
              [3,4]])

B = np.array([[2,1,1],
              [4,5,2]])

print("A: ",A.shape, " B: ",B.shape)
print(np.array(matrixMultiply(A,B)))
print(np.matmul(A,B))