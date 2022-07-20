import matplotlib.pyplot as plt
import cv2

img = cv2.imread("./images/img.png",cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

# fig, ax = plt.subplots()
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].set_facecolor("tan")
 
ax[0].imshow(img, cmap = 'gray')
ax[0].set_title('mask1', fontsize = 19)


plt.show()