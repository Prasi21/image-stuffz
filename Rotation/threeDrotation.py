from scipy.stats import truncnorm
import numpy as np  
import matplotlib.pyplot as plt
import cv2
import math

def rotateImage(src, alpha, beta, gamma, dx, dy, dz, f):
    """
    rotateImage rotates any input image by a given angle in the x, y and z planes
    :param src: the input image
    :param alpha: rotation around the x axis
    :param beta: rotation around the y axis
    :param gamma: rotation around the z axis (2d rotation)
    :param dx: translation around the axis
    :param dy: translation around the y axis
    :param dz: translation around the z axis (distance to image)
    :param f: focal distance (distance between camera and image)
    referenced from http://jepsonsblog.blogspot.com/2012/11/rotation-in-3d-using-opencvs.html
    """

    #convert to radians and start on x axis?
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)


    # get width and height for ease of use in matrices
    h, w = src.shape[:2]

    # Projection 2D -> 3D matrix
    A1 = np.array(
        [[1, 0, -w/2],
         [0, 1, -h/2],
         [0, 0, 1   ],
         [0, 0, 1   ]])

    
    # Rotation matrices around the X, Y, Z axis

    xa1 = math.cos(alpha)
    xa2 = math.sin(alpha)

    RX = np.array(
        [[1, 0,   0,    0],
         [0, xa1, -xa2, 0],
         [0, xa2, xa1,  0],
         [0, 0,   0,    1]])


    ya1 = math.cos(beta)
    ya2 = math.sin(beta)

    RY = np.array(
        [[ya1, 0, -ya2, 0],
         [0,   1, 0,    0],
         [ya2, 0, ya1,  0],
         [0,   0, 0,    1]])


    za1 = math.cos(gamma)
    za2 = math.sin(gamma)

    RZ = np.array(
        [[za1, -za2, 0, 0],
         [za2, za1,  0, 0],
         [0,   0,    1, 0],
         [0,   0,    0, 1]])


    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    #Translation Matrix
    T = np.array(
        [[1, 0, 0, dx],
         [0, 1, 0, dy],
         [0, 0, 1, dz],
         [0, 0, 0, 1 ]])

    # 3D -> 2D matrix
    A2 = np.array(
        [[f, 0, w/2, 0],
         [0, f, h/2, 0],
         [0, 0, 1,   0]])

    #Final tranformation matrix
    trans = np.dot(A2, np.dot(T, np.dot(R, A1)))

    # Apply matrix transformation
    dest = cv2.warpPerspective(src, M=trans, dsize=(w,h), flags=cv2.INTER_LANCZOS4)
    return dest

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# rotateImage(img, img, 0, 45, 0, 0, 0, 200, 200)
# rotateImage(img, img, -25, 125, 40, 0, 0, 200, 200)
img = cv2.imread('./images/9.png', cv2.IMREAD_UNCHANGED)
angle = np.zeros(3)
X = get_truncated_normal(mean=0, sd=30, low=-70, upp=70)
angle[0:2] = X.rvs(2)
Y = get_truncated_normal(mean=0, sd=90, low=-180, upp=180)
angle[2] = Y.rvs(1)

# angle[0], angle[1], angle[2] = 10, 10, 10

dest = rotateImage(img, angle[0], angle[1], angle[2], 0, 0, 400, 200)
cv2.imwrite("./images/9_3drotate.png", dest)
# cv2.imshow("Original Image", img)
# cv2.imshow("Rotated Image", dest)
cv2.waitKey(0)
