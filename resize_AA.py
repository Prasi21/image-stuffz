import numpy as np
import matplotlib as plt
import cv2 as cv


#Alpha mask of red channel
#scale highest value to 255 and multiply other values accordingly
#apply onto image?
#what happens to black parts?


img = cv.imread("./images/9.png")


print("Size: ", img.size)
print("Shape ", img.shape)



# inc_scale = 2
# dec_scale = 0.5

# large = cv.resize(img, (0,0), fx = inc_scale, fy=inc_scale, interpolation=cv.INTER_NEAREST)

# small = cv.resize(large, (0,0), fx = dec_scale, fy=dec_scale, interpolation=cv.INTER_AREA)


# cv.imwrite("./image stuffz/images/9_AA.png", small)




