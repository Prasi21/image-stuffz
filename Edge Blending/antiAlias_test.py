import numpy as np
import matplotlib as plt
import cv2 as cv


#scale factor
s = 10
thickness = 3

#create black image
img = np.zeros( (100*s,100*s, 3), np.uint8 )

#draw line
cv.line(img, (8*s, 8*s), (20*s, 25*s), (0,255,0), thickness)


#draw a rectangle
cv.rectangle(img, (10*s, 40*s), (60*s, 90*s), (255,0,0), thickness)

#draw a circle
cv.circle(img, (80*s, 20*s), 15*s, (0,0,255), thickness,cv.LINE_AA)


#draw a polygon
pts = np.array([ [50,10], [60,15], [75,30], [68,40], [55, 45] ])*s
#negative number in shapes is used to infer the missing length 
pts = pts.reshape((-1,1,2))
cv.polylines(img, [pts], True, (0,255,255), thickness)


#save image
savePath = "./image stuffz/AA_image.png"
cv.imwrite(savePath, img)
