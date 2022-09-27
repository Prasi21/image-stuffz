import numpy as np
import math
import cv2

# passed in (1500,1500)
#? tries to put higher prob in middle 1500?
def hc_probabilties_vector(map_size):
    map = np.ones(map_size)
    return map.flatten() / map.sum()

# Bbox is a dictionary containing the top left (xmin,ymin) and bottom right (xmax,ymax) coords of the image
#bboxes is a list of all the bboxes of all the signs placed so far
    #! Investigate
    # If the new right side is right of the old left side and
    # new left side is left of old right side, there is an x-intersection
    # repeat for y and if both true then there is an intersection 
def has_intersection(x0, y0, x1, y1, bboxes):
    for bbox in bboxes:
        if x1 > bbox['xmin'] and bbox['xmax'] > x0:
            if y1 > bbox['ymin'] and bbox['ymax'] > y0:
                return True
    return False

#! Investigate
# random number of flatten list of image shape modulus of image height give x coord (how far across that row)
# rand int of flattened list divided by image width gives image y (how many rows big)

def get_random_position(probabilities_vector, positions_list, img_size=(2048, 2048), sample_size=1):
    positions = np.random.choice(   
        positions_list, size=sample_size, p=probabilities_vector)
    positions = [(position % img_size[1], math.ceil(
        position / img_size[0]) - 1) for position in positions]

    return positions


# Template mask is template where alpha > 0 = 1 and alpha = 0 = 0
# returns tighter bbox with new coords
def remove_padding(template, template_mask):
    mask = template_mask[:, :, 0] > 0

    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return template[y0:y1, x0:x1], template_mask[y0:y1, x0:x1], x0, y0, x1, y1




def overlayMulti(target, bboxes):

    template = cv2.imread("./images/1.png")

    template_h, template_w, _ = template.shape
    target_h, target_w, _ = target.shape

    probabilities_vector = hc_probabilties_vector((1500, 1500))
    positions_list = np.arange(0, probabilities_vector.size)
    position = get_random_position(probabilities_vector, positions_list, target.shape[:-1])[0]
    print(f"Target shape: {target.shape}")
    print(f"Target shape slice: {target.shape[:-1]}")
    print(f"Position_List shape: {positions_list.shape}")
    print(positions_list)
    print(f"position: {position}")

    x0, y0 = position
    x1 = x0 + template_w
    y1 = y0 + template_h

    # Shift the image back and up if it is too far
    if x1 >= target_w:
            diff = x1 - target_w + 1
            x0 -= diff
            x1 -= diff
    if y1 >= target_h:
        diff = y1 - target_h + 1
        y0 -= diff
        y1 -= diff

    # Check for thin bg, giveup
    if x0 < 0 or y0 < 0 or x1 >= target_w or y1 >= target_h:
        return target, None#, scale, data

    # removed padding should be done before
    # Get the bbox of the sign and remove the padding of it

    # give up on overlaying
    if bboxes is not None and has_intersection(x0, y0, x1, y1, bboxes):
        return target, None#, scale, data

    # place template on target
    target[y0:y1, x0:x1] = template

    return target, {
        'xmin': x0,
        'ymin': y0,
        'xmax': x1,
        'ymax': y1
    } #info about transforms# , scale, data

    
if __name__ == "__main__":
    target = cv2.imread("./images/00014.png")
    bboxes = []

    total = 0   # signs placed so far
    nb_signs_in_img = np.random.randint(1, 5 + 1) # total signs to place

    while total < nb_signs_in_img:
        # target gets manipulated with each cycle of the overlay doesnt fail
        target, bbox = overlayMulti(target, bboxes)

        #? Store info about that sign in bbox where it can be saved in annotations?
        if bbox is not None:
            # bbox['category'] = template_category
            bboxes.append(bbox)
            # image_data['bbox_data'].append({
            #     'bbox': bbox,
            #     'data': data
            # })
            total += 1
        #! Randomly decide if you want another sign in a grid below it
        # probs = [0.4, 0.5]
        # while len(probs) > 0:
        #     do_place_below = np.random.choice([True, False], p=[probs[0], 1 - probs[0]]) and total < nb_signs_in_img
        #     if not (do_place_below and total < nb_signs_in_img and bbox):
        #         break
        #     probs = probs[1:]
        #     position = (bbox['xmin'], bbox['ymax'])



