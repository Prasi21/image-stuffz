import cv2
import math
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

def subplot(images):
    n = math.ceil(math.sqrt(len(images))) + 1
    fig, ax = plt.subplots(3, 4, figsize=(12, 12))
    x = 0

    try:
        for i in range(n):
            for j in range(n):
                ax[i][j].imshow(images[x][0], cmap = 'gray')
                ax[i][j].set_title(images[x][1], fontsize = 19)
                x += 1
    except IndexError:
        pass

    plt.show()


# https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
img = cv2.imread("./images/9.png")
# img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


imarray = np.random.rand(img.shape[0], img.shape[1], 1) * 255
# imarray = np.zeros(img.shape)
# imarray = np.ones(img.shape[0:2])*255
# imarray = np.random.uniform(0.5,1,img.shape[0:2])*255
im = imarray.astype('uint8')
im_grey = im
# im_grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(30,30))
# 
# dilate_grey = cv2.blur(im_grey,(7,7))
dilate_grey = cv2.GaussianBlur(im_grey,(3,3),0)
# dilate_grey = cv2.morphologyEx(im_grey, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)
# dilate_grey = cv2.dilate(im_grey, kernel, iterations=1)
# dilate_grey = im_grey
dilate_grey = cv2.cvtColor(dilate_grey, cv2.COLOR_GRAY2BGR)



hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

lower_red = np.array([160, 100, 50])
upper_red = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

mask_inv = cv2.bitwise_not(mask)

res = cv2.bitwise_and(img, img, mask=mask)

##########
# res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
# res[:,:,2] = res[:,:,2] + 50
slice_rdm = dilate_grey[np.where(res[:,:,2] != 0)]
slice = res[np.where(res[:,:,2] != 0)]
slice = cv2.addWeighted(slice, 0.5, slice_rdm, 0.5,0)
# # slice = (255-slice)* (slice_rdm/255) + slice
# print(slice_rdm[:,2])
# slice[:,2] = slice[:,2] + slice_rdm[:,2]
# slice[np.where(slice>255)] = 255
res[np.where(res[:,:,2] != 0)] = slice 
# res[np.where(res[:,:,2] != 0)] = ((255 - res[np.where(res[:,:,2] != 0)])).astype(np.unit8)
# res = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
##########

background = cv2.bitwise_and(grey, grey, mask=mask_inv)

background = np.stack((background,)*3, axis=-1)

added_img = cv2.add(res,background)


images = [img, grey, mask, res, background, im, im_grey, dilate_grey, added_img]
names = ["Original", "Grey", "Mask", "Res", "Background", "Random Img", "Random Grey", "Dilated Grey", "Final"]
img_and_name = [(img,"Original"), (grey,"Grey"), (mask, "Mask"), (res, "Res"), (background, "Background"), (im, "Random Img"), 
            (im_grey, "Random Grey"), (dilate_grey, "Dilated Grey"), (added_img, "Final")]

subplot(img_and_name)
