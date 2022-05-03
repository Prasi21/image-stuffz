import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys

def colour_slicer(channel,  img):
    colour = img.copy()
    colour[:,:,(channel + 1)%3::3] = 0
    colour[:,:,(channel + 2)%3::3] = 0
    return colour


img = cv.imread("c:/Users/prasi/OneDrive/Documents/Code Practice/image stuffz/apple.jpg")

#converting the original image from RGB to BGR
temp = img.copy()
img[:,:,0::3] = temp[:,:,2::3] 
img[:,:,2::3] = temp[:,:,1::3]


red = colour_slicer(2,img)
green = colour_slicer(1,img)
blue = colour_slicer(0,img)

f, grid = plt.subplots(2,2)
grid[0,0].imshow(img)
grid[0,0].title.set_text("Apple")


grid[0,1].imshow(red)
grid[0,1].title.set_text("Red Apple")

grid[1,0].imshow(green)
grid[1,0].title.set_text("Blue Apple")

grid[1,1].imshow(blue)
grid[1,1].title.set_text("Green Apple")

plt.tight_layout()
plt.show()


# cv.imshow('apple.jpg')


# if img is None:
#     sys.exit("Could not read the image")



# cv.imshow("Display window", img)
# k = cv.waitKey(0)

# if k == ord("s"):
#     cv.imwrite("apple.png")