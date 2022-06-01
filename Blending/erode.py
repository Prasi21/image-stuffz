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


template = cv2.imread("./images/9.png",cv2.IMREAD_UNCHANGED)#.astype(np.float32) / 255.0
target = cv2.imread("./images/00014.png")#.astype(np.float32) / 255.0
# target, add_value, multiply_value = augment_target(target)

# mask = get_mask_from_image(template)

# My method of creating a mask
ret, mask1 = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)    
mask = np.ones((200,200,3),dtype=np.uint8)
mask = cv2.bitwise_and(mask, mask, mask=mask1)


steps = 3



# mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)) #? do we need this
blend_mask = mask.astype(np.float32) * (1.0 / steps)
kernel = np.ones((3, 3), np.uint8)


for step in range(steps - 1):
    mask = cv2.erode(mask, kernel)
    blend_mask += mask * (1.0 / steps)

# blend_mask = blend_mask[1:-1, 1:-1] #? along with this?

x0 = 500
y0 = 500
x1 = x0 + template.shape[1]
y1 = y0 + template.shape[1]


blended = (target[y0:y1, x0:x1] * (1 - blend_mask)) + (template[:, :, [0, 1, 2]] * blend_mask)
target[y0:y1, x0:x1] = blended
blended = blended.astype(np.float32) / 255.0 #uneeded when placed on to bg


cv2.imshow("bgfull",target)
cv2.imshow("Original",template)


cv2.waitKey(0)
