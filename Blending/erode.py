import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from imgaug import augmenters as iaa


# Function used by Tabelini to apply colour to the entire image
def augment_target(target, multiply_value=None, add_value=None):
    if add_value is None:
        add_value = float(np.random.uniform(-120, 120))
    if multiply_value is None:
        multiply_value = float(np.random.uniform(0.75, 1.25))

    seq = iaa.Sequential([
        iaa.Add((add_value, add_value)),
        iaa.Multiply((multiply_value, multiply_value)),
    ])
    target = (target * 255.0).astype(np.uint8)
    return (seq.augment_image(target) / 255.0).astype(np.float32), add_value, multiply_value




# Function used by me to check if methods of creating the mask produce the same result
# Compares two images to see if they are identical
def compareImage(original, duplicate):
    if original.shape == duplicate.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The images are completely Equal")
    else:
        print("different shapes")

#threshold method achives same in less lines
def get_mask_from_image(alpha_image):
    alpha_channel = alpha_image[:, :, -1]
    mask = np.zeros_like(alpha_image[:, :, :-1])
    mask[:, :, 0][alpha_channel > 0] = 1
    mask[:, :, 1][alpha_channel > 0] = 1
    mask[:, :, 2][alpha_channel > 0] = 1

    return mask

fig, ax = plt.subplots(3, 3, figsize=(12, 8))


template = cv2.imread("./images/1.png",cv2.IMREAD_UNCHANGED)#.astype(np.float32) / 255.0
# plt.figure(1)
# plt.imshow(template)
template = cv2.imread("./images/bullets.png",cv2.IMREAD_UNCHANGED)
# plt.figure(2)
# plt.imshow(template)
target = cv2.imread("./images/00014.png")#.astype(np.float32) / 255.0



ax[2][1].imshow(template, cmap = 'gray')
ax[2][1].set_title('240', fontsize = 19)
ax[2][2].imshow(template, cmap = 'gray')
ax[2][2].set_title('resized', fontsize = 19)


# My method of creating a mask
ret, mask1 = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)    
mask = np.ones((template.shape[0:2] + (3,) ),dtype=np.uint8)
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

ax[1][1].imshow(blend_mask, cmap = 'gray')
ax[1][1].set_title('blend_mask', fontsize = 19)

new_size = 480
template = cv2.resize(template, (new_size, new_size))
blend_mask = cv2.resize(blend_mask, (new_size, new_size))


x0 = 300
y0 = 300
x1 = x0 + template.shape[1]
y1 = y0 + template.shape[1]


ax[2][0].imshow(cv2.cvtColor(template[:, :, [0, 1, 2]].astype(np.float32)/255 * blend_mask, cv2.COLOR_RGB2BGR), cmap = 'gray')
ax[2][0].set_title('blended template', fontsize = 19)

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