import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2

# https://analyticsindiamag.com/complete-tutorial-on-image-transformations-with-opencv/
image = cv2.imread("./images/9.png", cv2.IMREAD_UNCHANGED)


pad = 0
image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
num_rows, num_cols = image.shape[:2]

x = 0.33
y = 0.66
src_points = np.float32([[0,0], [num_cols-1-pad,0], [0,num_rows-1-pad], [num_cols-1-pad ,num_rows-1-pad]])
dst_points = np.float32([[0,0], [num_cols-1-pad,0], [int(x*(num_cols-pad)),num_rows-1-pad], [int(y*(num_cols-pad)),num_rows-1-pad]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_protran = cv2.warpPerspective(image, projective_matrix, (num_cols+0,num_rows+0))

result = img_protran

cv2.imwrite("C:/Users/prasi/Documents/Code Practice/image stuffz/images/9_warped.png",result)
cv2.imshow("rotated", result)
cv2.waitKey(0)
cv2.destroyAllWindows()