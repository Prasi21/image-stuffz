import matplotlib.pyplot as plt
import cv2
import math

def subplot(images, names):
    n = math.ceil(math.sqrt(len(images)))
    fig, ax = plt.subplots(n, n, figsize=(12, 12))
    x = 0

    try:
        for i in range(n):
            for j in range(n):
                ax[i][j].imshow(images[x], cmap = 'gray')
                ax[i][j].set_title(names[x], fontsize = 19)
                x += 1
    except IndexError:
        pass

    plt.show()

# For an array containing tuples of (image, name)
def subplot(image_and_name):
    n = math.ceil(math.sqrt(len(image_and_name)))
    fig, ax = plt.subplots(n, n, figsize=(12, 12))
    x = 0

    try:
        for i in range(n):
            for j in range(n):
                ax[i][j].imshow(image_and_name[x][0], cmap = 'gray')
                ax[i][j].set_title(image_and_name[x][1], fontsize = 19)
                x += 1
    except IndexError:
        pass

    plt.show()


img = cv2.imread("./images/faded_sticker.png")
print(len(cv2.split(img)))
print(img.shape[-1])
new = img
if(img.shape[-1] == 3):
    new = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
cv2.imwrite("./images/faded_sticker2.png", new)

# img1 = cv2.imread("./images/oldGrey.png",cv2.IMREAD_UNCHANGED)
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGRA)
# img2 = cv2.imread("./images/newGrey.png", cv2.IMREAD_UNCHANGED)
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGRA)

# fig, ax = plt.subplots()
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].set_facecolor("tan")
 
# ax[0].imshow(img1, cmap = 'gray')
# ax[0].set_title('Old Grey', fontsize = 19)
# ax[1].imshow(img2, cmap = 'gray')
# ax[1].set_title('New Grey', fontsize = 19)

# plt.show()