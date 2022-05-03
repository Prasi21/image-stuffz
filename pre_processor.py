import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# img_height, img_width = 300, 300
# n_channels = 4
# transparent_img = np.zeros((img_height, img_width, n_channels))

# cv.imwrite("./image stuffz/transparent_img.png", transparent_img)

sign = cv.imread("./image stuffz/STOP_SIGN-1.jpg")

# temp = sign.copy()
# sign[:,:,0::3] = temp[:,:,2::3] 
# sign[:,:,2::3] = temp[:,:,1::3]

rgba = cv.cvtColor(sign, cv.COLOR_RGB2RGBA)

rgba[:, :, 3] = 255

boundries = np.zeros( (len(rgba),2), dtype=int)

for r in range(0,1600):
    # print("r: ",r)
    for c in range(0,1200):
        # print("row: ",r," Lcolumn: ",c)
        # print("r: ",r," c: ",c,rgba[r,c,0])
        if( (rgba[r,c] == [255,255,255,255]).all() == False):
            boundries[r,0] = c
            break
    for c in range(1199,-1,-1):
        # print("row: ",r," Rcolumn: ",c)
        if( (rgba[r,c] == [255,255,255,255]).all() == False):
            boundries[r,1] = c
            break

# r = 0
# for edge in boundries:
#     rgba[r,0:edge[0]:1,3] = 0
#     rgba[r,255:edge[1]:-1,3] = 0
#     r+=1



for r in range(len(boundries)):
    # print(boundries[r])
    lSlice = slice(0,boundries[r][0])
    rSlice = slice(boundries[r][1],256)
    rgba[r,lSlice,3] = 0
    rgba[r,rSlice,3] = 0



print(sign.shape)
print("RGBA shape:", rgba.shape)
# sign = sign[:, :, np.newaxis]

cv.imwrite("./image stuffz/transparent_img.png", rgba)

plt.imshow(rgba)
plt.show()



