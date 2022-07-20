import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from imgaug import augmenters as iaa
from skimage.io import imread, imshow
from skimage.morphology import erosion, dilation


def multi_erosion(img, kernel, iterations):
    for x in range(iterations):
        img = cv2.erode(img, kernel)
    return img

def multi_dilate(img, kernel, iterations):
    for x in range(iterations):
        img = cv2.dilate(img, kernel)
    return img


fig, ax = plt.subplots(2, 3, figsize=(12, 5))

template = cv2.imread("./images/9.png",cv2.IMREAD_UNCHANGED)
template = cv2.cvtColor(template, cv2.COLOR_RGBA2BGRA)
ret, mask1 = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)    
mask = np.ones((template.shape[0:2] + (3,) ),dtype=np.uint8)*255
mask = cv2.bitwise_and(mask, mask, mask=mask1)


ax[1][1].imshow(mask1, cmap = 'gray')
ax[1][1].set_title('mask1', fontsize = 19)

ax[1][2].imshow(mask, cmap = 'gray')
ax[1][2].set_title('mask', fontsize = 19)


color = (255,255,255)

circle_img = np.zeros((200,200))
circle_img= cv2.circle(circle_img, (100, 100), 50, color, -1)

img1 = mask

ax[0][0].imshow(img1, cmap = 'gray')
ax[0][0].set_title('Original', fontsize = 19)


kernel = np.ones((3,3))
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

ax[0][1].imshow(cross, cmap = 'gray')
ax[0][1].set_title('Cross', fontsize = 19)

# eroded = multi_erosion(img1, cross, 5)
# eroded = cv2.erode(img1, cross, iterations=3)
# eroded = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel=cross)
eroded = cv2.erode(img1, cross, iterations=5)

ax[0][2].imshow(eroded, cmap = 'gray')
ax[0][2].set_title('eroded', fontsize = 19)

template[:,:,3] = eroded[:,:,1]

ax[1][0].imshow(template, cmap = 'gray')
ax[1][0].set_title('Template final', fontsize = 19)

plt.show()

template = cv2.cvtColor(template, cv2.COLOR_BGRA2RGBA)
cv2.imwrite("./images/clean.png", template)
