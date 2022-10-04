import cv2
import math
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def main():
    # https://medium.com/featurepreneur/colour-filtering-and-colour-pop-effects-using-opencv-python-3ce7d4576140
    sign = cv2.imread("./images/9.png", cv2.IMREAD_UNCHANGED)
    img = sign.copy()
    # Search for csgo stash graffiti with transparent and size icon
    sticker = cv2.imread("./images/sticker4.png", cv2.IMREAD_UNCHANGED)
    if len(cv2.split(sticker)) == 3:
        sticker = cv2.cvtColor(sticker, cv2.COLOR_RGB2RGBA)
        sticker[:, :, 3] = 255  # Keep it opaque

    scale_heigth = 100
    scale_percent = scale_heigth/sticker.shape[1] # percent of original size
    width = int(sticker.shape[1] * scale_percent)
    height = int(sticker.shape[0] * scale_percent)
    dim = (width, height)
    # dim = (50,50)
    
    sticker_sml = cv2.resize(sticker, dim, cv2.INTER_AREA)

    bg_x, bg_y = img.shape[0:2]
    fg_x, fg_y = sticker_sml.shape[0:2]
    X_norm = get_truncated_normal(mean=bg_x/2 - fg_x, sd=15, low=0, upp=bg_x - fg_x)
    Y_norm = get_truncated_normal(mean=bg_y/2 - fg_y, sd=30, low=0, upp=bg_y - fg_y)
    x1 = int(X_norm.rvs(1))
    x2 = x1 + fg_x
    y1 = int(Y_norm.rvs(1))
    y2 = y1 + fg_y

    print(f"Top Left: {x1}, {y1}  Bottom Right: {x2}, {y2}")

    slice = img[x1:x2, y1:y2,0:3]
    slice[np.where(sticker_sml[:,:,3] != 0)] = sticker_sml[:,:,0:3][np.where(sticker_sml[:,:,3] != 0)]


    # slice = np.where(sticker[:,:,3] == 0)
    sign = cv2.cvtColor(sign, cv2.COLOR_RGBA2BGRA)
    sticker = cv2.cvtColor(sticker, cv2.COLOR_RGBA2BGRA)
    sticker_sml = cv2.cvtColor(sticker_sml, cv2.COLOR_RGBA2BGRA)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    image_and_name = [(sign, "Original Sign"), (sticker, "Sticker"), (sticker_sml, "Small Sticker"), (img, "Final")]
    subplot(image_and_name)



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

    
    

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


if __name__ == "__main__":
    main()

