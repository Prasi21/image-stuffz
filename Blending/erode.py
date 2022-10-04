import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from imgaug import augmenters as iaa

#todo fix bullet holes

fig, ax = plt.subplots(3, 3, figsize=(12, 8))


template = cv2.imread("./images/9.png",cv2.IMREAD_UNCHANGED)#.astype(np.float32) / 255.0
template = cv2.imread("./images/rotating.png",cv2.IMREAD_UNCHANGED)
target = cv2.imread("./images/00014.png",cv2.IMREAD_UNCHANGED)#.astype(np.float32) / 255.0

# if len(cv2.split(target)) == 3:
#     target = cv2.cvtColor(target, cv2.COLOR_RGB2RGBA)
#     target[:, :, 3] = 255


ax[2][1].imshow(cv2.cvtColor(target, cv2.COLOR_RGBA2BGRA), cmap = 'gray')
ax[2][1].set_title('original', fontsize = 19)
ax[2][2].imshow(cv2.cvtColor(template, cv2.COLOR_RGBA2BGRA), cmap = 'gray')
ax[2][2].set_title('resized', fontsize = 19)


#TODO is gaussian blur better than overdoing erodes
# My method of creating a mask
# ret, mask1 = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)    
ret, mask1 = cv2.threshold(cv2.GaussianBlur(template[:, :, 3], (5,5), 0), 0, 255, cv2.THRESH_BINARY)    
mask = np.ones( (template.shape[0], template.shape[1], 3),dtype=np.uint8)
mask = cv2.bitwise_and(mask, mask, mask=mask1)

ax[0][1].imshow(mask*255, cmap = 'gray')
ax[0][1].set_title('Original Mask', fontsize = 19)


steps = 3


# mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #? do we need this
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

pad = 20
mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

mask = cv2.erode(mask, cross, iterations=2)
# mask = cv2.GaussianBlur(mask, (5,5), 0)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=ellipse, iterations=5)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=ellipse, iterations=5)
mask = mask[pad:-pad, pad:-pad]

# mask = cv2.dilate(mask, kernel, iterations=3)
ax[0][2].imshow(mask*255, cmap = 'gray')
ax[0][2].set_title('Closed mask', fontsize = 19)


blend_mask = mask.astype(np.float32) * (1.0 / steps)
for step in range(steps - 1):
    mask = cv2.erode(mask, cross)
    blend_mask += mask * (1.0 / steps)
# blend_mask = blend_mask[1:-1, 1:-1] #? along with this?
ax[1][0].imshow(mask*255, cmap = 'gray')
ax[1][0].set_title('eroded mask', fontsize = 19)

#some kind of bitwise and: blend mask is white (255,255,255) and template alpha = 255 makes blend_mask black (0,0,0)

d = (blend_mask[:,:,0] == 255)# + (template[:,:,3] == 0)
d = d[..., np.newaxis]
temp2 = np.zeros((template.shape[0:2]+ (3,)))
temp2 = np.stack((template[:,:,3].astype(np.float32) / 255.0,),axis=-1)
# temp2[:,:,0] = template[:,:,3].astype(np.float32) / 255.0
# temp2[:,:,1] = template[:,:,3].astype(np.float32) / 255.0
# temp2[:,:,2] = template[:,:,3].astype(np.float32) / 255.0

print(f"d:{d.shape}")
print(f"dshape:{blend_mask.shape} dsize{blend_mask.size}")
blend_mask = np.where( blend_mask == (1,1,1), temp2, blend_mask)
print(f"dshape:{blend_mask.shape} dsize{blend_mask.size}")


new_size = 480
template = cv2.resize(template, (new_size, new_size))
blend_mask = cv2.resize(blend_mask, (new_size, new_size))
ax[1][1].imshow(blend_mask, cmap = 'gray')
ax[1][1].set_title('blend_mask', fontsize = 19)
ax[2][0].imshow(temp2, cmap = 'gray')
ax[2][0].set_title('temp2', fontsize = 19)


# plt.show()

x0 = 300
y0 = 300
x1 = x0 + template.shape[1]
y1 = y0 + template.shape[1]


# ax[2][0].imshow(cv2.cvtColor(template[:, :, [0, 1, 2]].astype(np.float32)/255 * blend_mask, cv2.COLOR_RGB2BGR), cmap = 'gray')
# ax[2][0].set_title('blended template', fontsize = 19)

# blended = np.zeros((template.shape))
blended = (target[y0:y1, x0:x1, [0, 1, 2]] * (1 - blend_mask)) + (template[:, :, [0, 1, 2]] * blend_mask)
target[y0:y1, x0:x1, [0, 1, 2]] = blended
blended = blended.astype(np.float32) / 255.0 #uneeded when placed on to bg
# blended = blended *255
target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
cv2.imwrite("./images/full.png", target)


template = cv2.cvtColor(template, cv2.COLOR_RGBA2BGRA)
blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
ax[0][0].set_facecolor("tan")
ax[0][0].imshow(template, cmap = 'gray')
ax[0][0].set_title('original', fontsize = 19)

ax[1][2].imshow(blended, cmap = 'gray')
ax[1][2].set_title('blended', fontsize = 19)

fig, ax = plt.subplots()
ax.set_facecolor('#0F0F0F0F')
im = ax.imshow(target)
plt.show()