import time
import numpy as np
import matplotlib as plt
import cv2
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



def main():

    x0= 0
    y0= 0
    x1= 200
    y1= 200
    
    template = cv2.imread("../images/9.png", cv2.IMREAD_UNCHANGED)
    target = cv2.imread("../images/00014.png", cv2.IMREAD_UNCHANGED)
    template_mask = template[1]

    print(template.shape)
    print(target.shape)

    template = blend(template, template_mask, target, {
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1
    })
    # place template on target
    target[y0:y1, x0:x1] = template

    return target, {
        'xmin': x0,
        'ymin': y0,
        'xmax': x1,
        'ymax': y1
    }#, scale, data





def blend(template, template_mask, target_image, target_bbox, steps=3):
    template = (template * 255).astype(np.uint8)
    target_image = (target_image * 255).astype(np.uint8)

    temp_template_mask = template_mask.copy()
    temp_template_mask = cv2.copyMakeBorder(temp_template_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    blend_mask = temp_template_mask.astype(np.float32) * (1.0 / steps)
    kernel = np.ones((3, 3), np.uint8)

    for step in range(steps - 1):
        temp_template_mask = cv2.erode(temp_template_mask, kernel)
        blend_mask += temp_template_mask * (1.0 / steps)

    x0 = target_bbox['x0']
    y0 = target_bbox['y0']
    x1 = target_bbox['x1']
    y1 = target_bbox['y1']
    blend_mask = blend_mask[1:-1, 1:-1]
    blended = (target_image[y0:y1, x0:x1] * (1 - blend_mask)) + (template[:, :, [0, 1, 2]] * blend_mask)

    print("ayo")

    return blended.astype(np.float32) / 255.0




def timer(func):
    def wrapper():
        # start = datetime.datetime.now()
        start = time.time()
        func()
        runtime = time.time() - start
        print(runtime)
    return wrapper

#use @time before a function declaration to time it


if __name__ == "__main__":
    main()