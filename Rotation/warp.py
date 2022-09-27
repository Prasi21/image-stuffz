import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import math
from skimage import data, transform

def rotate():
    # img = data.camera()
    img = cv2.imread("./images/9.png", cv2.IMREAD_UNCHANGED)

    theta = np.deg2rad(10)
    tx = 0
    ty = 0

    S, C = np.sin(theta), np.cos(theta)

    # Rotation matrix, angle theta, translation tx, ty
    H = np.array([[C, -S, tx],
                [S,  C, ty],
                [0,  0, 1]])

    # Translation matrix to shift the image center to the origin
    r, c = 250, 250 #img.shape
    T = np.array([[1, 0, -c / 2.],
                [0, 1, -r / 2.],
                [0, 0, 1]])

    # Skew, for perspective
    S = np.array([[1, 0, 0],
                [0, 1.3, 0],
                [0, 1e-3, 1]])

    img_rot = transform.homography(img, H)
    img_rot_center_skew = transform.homography(img, S.dot(np.linalg.inv(T).dot(H).dot(T)))

    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax1.imshow(img_rot, cmap=plt.cm.gray, interpolation='nearest')
    ax2.imshow(img_rot_center_skew, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()


# img = cv2.imread("./Edge Blending/apple.jpg")

img = cv2.imread("./images/9.png", cv2.IMREAD_UNCHANGED)

print(img.shape)

img_copy = np.copy(img)

img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(img_copy)

def onclick(event):
    if event.xdata != None and event.ydata != None:
        print(event.xdata, event.ydata)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# https://www.thepythoncode.com/article/image-transformations-using-opencv-in-python
rows, cols, dim = img.shape

M = np.float32([[1, 0, 50],
                [0, 1, 50],
                [0, 0, 1]])

translated_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
translated_img = cv2.warpPerspective(img, M, (cols, rows))



plt.imshow(translated_img)
plt.show()


rotate()


# plt.imshow(img_copy)
# plot = plt.imshow(img)
# plt.show()


pt_A = [0, 0]
pt_B = [0, 200]
pt_C = [200, 200]
pt_D = [200, 0]

out_A =  [0, 0]
out_B =  [30, 200]
out_C =  [100, 200]
out_D =  [100, 0]

out_A =  [0, 0]
out_B =  [0, 200]
out_C =  [200 * 0.75, 200 - 50]
out_D =  [200 * 0.75, 80]


input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([out_A, out_B, out_C, out_D])

# Here, I have used L2 norm. You can use L1 also.
width_AD = np.sqrt(((out_A[0] - out_D[0]) ** 2) + ((out_A[1] - out_D[1]) ** 2))
width_BC = np.sqrt(((out_B[0] - out_C[0]) ** 2) + ((out_B[1] - out_C[1]) ** 2))
maxWidth = max(int(width_AD), int(width_BC))

height_AB = np.sqrt(((out_A[0] - out_B[0]) ** 2) + ((out_A[1] - out_B[1]) ** 2))
height_CD = np.sqrt(((out_C[0] - out_D[0]) ** 2) + ((out_C[1] - out_D[1]) ** 2))
maxHeight = max(int(height_AB), int(height_CD))

# Compute the perspective transform M
# M = cv2.getPerspectiveTransform(input_pts,output_pts)

# maxHeight = 200
# maxWidth = 200
a = 20

# T = np.array([[1  0  -image_width/2]
#             [0  1  -image_height/2]
#             [0  0   1]

# R = np.array([[math.cos(a),-math.sin(a),0],
#      [math.sin(a),math.cos(a),0],
#      [0,0,0]])

# M = R.astype(int)
# print(M)
# out = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

# out_copy = cv2.cvtColor(out,cv2.COLOR_BGR2RGB)


# plt.imshow(out_copy)
# plt.show()


