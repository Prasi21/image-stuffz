from cv2 import IMREAD_UNCHANGED
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#IMREAD_UNCHANGED makes sure the alpha channel is read in
sign = cv.imread("./images/9.png", IMREAD_UNCHANGED)
cv.waitKey(0)
blurred_img = cv.blur(sign,(25,25))
# blurred_img = cv.GaussianBlur(sign, (25, 25), 0)
mask = np.zeros(sign.shape, np.uint8)


#greyscale
grey = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)

#? thresh = cv.threshold(grey, 60, 255, cv.THRESH_BINARY)[2]
#? contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# #find canny edges
edged = cv.Canny(grey, 30, 200)

# #finding contours
# #use a copy edged.copy() because finding contours edits image
contours, hierarchy, = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.imshow("Canny edges after contouring", edged)
cv.waitKey(0)


# draw all contours
# -1 signifies all contours
cv.drawContours(mask, contours, -1, (255,255,255, 255), 2)

cv.imshow("contours", mask)
cv.waitKey(0)

#replace the white mask with the edge of the blurred image
output = cv.bitwise_and(mask, blurred_img)

#alpha channel becomes and average of the colour channels to copy the blur
output[:,:,3] = (output[:,:,0] + output[:,:,1] + output[:,:,2])/3

r,g,b,a = cv.split(output)

output = np.where(output==0, sign, output)
#equivalent to:
#? for i in range(0,4):
#?     output[:,:,i] = np.where(a==0,sign[:,:,i],output[:,:,i])


cv.imshow("blur edges", output)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("C:/Users/prasi/OneDrive/Documents/Code Practice/image stuffz/images/9_blurred.png",output)






