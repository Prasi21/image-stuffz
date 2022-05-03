import numpy as np
import matplotlib as plt
import cv2 
import os
import math
from PIL import Image, ImageOps
import glob

def create_alpha(img, alpha_channel):
    """Returns an alpha channel that matches the white background.\n
    Based on Alexandros Stergiou's find_borders() function."""

    # Read and decode the image contents for pixel access
    pix = img.load()

    # Note PIL indexes [x,y] while OpenCV indexes [y,x]
    # alpha_channel must be indexed [y,x] to merge with OpenCV channels later

    min = 200
    width, height = img.size
    # Loop through each row of the image
    for y in range(0, height):
        # First loop left to right
        for x in range(0, width, 1):
            # Retrieve a tuple with RGB values for this pixel
            rgb = pix[x,y]
            # Make transparent if the pixel is white (or light enough)
            if rgb[0] >= min and rgb[1] >= min and rgb[2] >= min:
                alpha_channel[y,x] = 0
            # If pixel is not white then we've hit the sign so break out of loop
            else:
                break

        # Then loop backwards, right to left
        for x in range(width-1, -1, -1):
            rgb = pix[x,y]
            if rgb[0] >= min and rgb[1] >= min and rgb[2] >= min:
                alpha_channel[y,x] = 0
            else:
                break

    return alpha_channel


def delete_background(image_path, save_path):
    """Deletes the white background from the original sign.\n
    Based on Alexandros Stergiou's manipulate_images() function."""
    # Open the image using PIL (don't read contents yet)
    img = Image.open(image_path)
    img = img.convert('RGB')  # TODO: Does this have any effect??

    # Open the image again using OpenCV and split into its channels
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    channels = cv2.split(image)

    # Create a fully opaque alpha channel, same dimensions and dtype as the image
    # create_alpha() modifies it to make the white background transparent
    alpha_channel = np.ones(channels[0].shape, dtype=channels[0].dtype) * 255
    alpha_channel = create_alpha(img, alpha_channel)

    # Merge alpha channel into original image
    image_RGBA = cv2.merge((channels[0], channels[1], channels[2], alpha_channel))

    # Mask the image so that deleted, invisible background pixels are black instead of white
    # This is crucial to accurate damage values, as damage functions also use this masking function and will turn those
    # pixels black anyway, artificially inflating all damage values when using the structural similarity measure
    image_RGBA = cv2.bitwise_and(image_RGBA, image_RGBA, mask=alpha_channel)

    cv2.imwrite(save_path, image_RGBA)
    img.close()


delete_background("./image stuffz/STOP_SIGN-1.jpg","./image stuffz/clean_image.jpg")


# # root_dir needs a trailing slash (i.e. /root/dir/)
# for filename in glob.iglob('/./' + '**/*.txt', recursive=True):
#     delete_background(filename,filename + ")