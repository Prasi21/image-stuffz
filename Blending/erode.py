import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
from imgaug import augmenters as iaa


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



template = cv2.imread("./images/1.png",cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
target = cv2.imread("./images/00014.png").astype(np.float32) / 255.0
# target, add_value, multiply_value = augment_target(target)


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

mask = get_mask_from_image(template)

# ret, mask3 = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)    
# mask1 = np.ones((200,200,3),dtype=np.uint8)
# mask3 = cv2.bitwise_and(mask1, mask1, mask=mask3)
# print("mask 3: ",mask3[0,0,0].type)
# print("mask 1: ",mask[0,0,0].type)


# compareImage(mask, mask3)

# np.set_printoptions(threshold=np.inf)
# print(mask)

# cv2.imshow("mask",mask)
# cv2.imshow("mask3",mask3)
# cv2.imshow("bg",target)
cv2.imshow("original",template)

steps = 3

template = (template*255).astype(np.uint8)
target = (target * 255).astype(np.uint8)

mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

blend_mask = mask.astype(np.float32) * (1.0 / steps)
kernel = np.ones((3, 3), np.uint8)


for step in range(steps - 1):
    mask = cv2.erode(mask, kernel)
    blend_mask += mask * (1.0 / steps)

# cv2.imshow("blend mask",blend_mask)

x0 = 500
y0 = 500
x1 = x0 + template.shape[1]
y1 = y0 + template.shape[1]
blend_mask = blend_mask[1:-1, 1:-1]
print("template: ",template.shape,"\nmask: ",blend_mask.shape)
# cv2.imshow("bg",target[y0:y1, x0:x1])
cv2.imshow("bgfull",target)


blended = (target[y0:y1, x0:x1] * (1 - blend_mask)) + (template[:, :, [0, 1, 2]] * blend_mask)
blended = blended.astype(np.float32) / 255.0


# print("blend mask",blend_mask.shape())
# blended = cv2.bitwise_and(blend_mask, )
cv2.imshow("blended",blended)

cv2.waitKey(0)
