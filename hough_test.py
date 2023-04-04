import math
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

points = np.load('coords2.npy')

setX = abs(616 - max(elem[0] for elem in points))
setY = abs(616 - max(elem[1] for elem in points))
def make_image(points):
    map_size = max((max(elem[1] for elem in points) - min(elem[0] for elem in points)),
        (max(elem[0] for elem in points) - min(elem[0] for elem in points)))
    map = [[0 for i in range(map_size + 10)] for j in range(map_size + 10)]
    for elem in points:
        map[elem[0] - setX][elem[1] - setY] = 255
    return np.array(map)

image = make_image(points)
image8 = np.asarray(image, dtype=np.uint8)
lines = cv2.HoughLinesP(image8, 1, np.pi / 180, 50, None, 50, 10)
for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(image8, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", image8)
cv2.waitKey(0)


