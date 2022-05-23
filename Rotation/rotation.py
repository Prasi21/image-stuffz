import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import random

def timer(func):
    def wrapper():
        # start = datetime.datetime.now()
        start = time.time()
        func()
        runtime = time.time() - start
        print(runtime)
    return wrapper

#use @time before a function declaration to time it

# @timer
def main():
    image = cv2.imread("./images/9.png", cv2.IMREAD_UNCHANGED)

    pad = 40
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
    if(np.random.randint(10) < 10): # 50% chance of rotating
        angle = int(np.random.normal(0,1)*180)
        cv2.imshow("padded", image)
        cv2.waitKey(0)
        print("Padded Image shape: ",image.shape)
        print("angle: ",angle)

        # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point  
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    else:
        result = image

    cv2.imwrite("C:/Users/prasi/Documents/Code Practice/image stuffz/images/9_rotated.png",result)
    cv2.imshow("rotated", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()

