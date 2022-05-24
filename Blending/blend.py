import time
import numpy as np
import matplotlib as plt
import cv2


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