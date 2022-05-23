import cv2
import matplotlib as plt
import numpy as np
import math

img = cv2.imread('./images/apple.jpg')

#getting the first two elements of the shape
height, width = img.shape[:2]

#coordinates of centre of image
centre = (width/2, height/2)

rotate_matrix = cv2.getRotationMatrix2D(center=centre, angle=45, scale=1)

scale = 1
a = 45 #angle in radians
a = math.radians(a) 

a1 = math.cos(a) * scale
a2 = math.sin(a) * scale

# Rotation Matrix
R = np.array([[a1,  a2, (1-a1) * centre[0] - a2 * centre[1] ],
              [-a2, a1, a2 * centre[0] + (1-a1) * centre[1] ]], 
              dtype=float)

print("R:\n",R)


tx, ty = 50, 50
# Translation Matrix
T = np.array([[0, 0, tx],
              [0, 0, ty]],
              dtype=float)

R = R + T
print("R+T:\n",R)


print("Original shape: ",img.shape)
print("Transformation matrix:\n", rotate_matrix)

# Apply Rotation
rotate_img = cv2.warpAffine(src=img, M=R, dsize=(width,height))
# Apply Translation
# rotate_img = cv2.warpAffine(src=rotate_img, M=T, dsize=(width,height))


print("New shape: ",rotate_img.shape)

cv2.imshow("Original Image", img)
cv2.imshow("Rotated Image", rotate_img)
cv2.waitKey(0)


