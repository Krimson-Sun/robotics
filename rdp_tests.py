from coords_parsing import get_decart_coords
from rdp import rdp
import matplotlib.pyplot as plt
import numpy as np
import math

walls = []
obst = []
points = rdp(np.array(get_decart_coords('examp12.txt')), epsilon=0.15, algo='iter')


flag = True
for i in range(len(points) - 1):
    if math.sqrt((points[i][0] - points[i + 1][0]) ** 2 + (points[i][1] - points[i + 1][1]) ** 2) < 0.3:
        if flag:
            walls.append(points[i])
        else:
            obst.append(points[i])
    else:
        if flag:
            obst.append(points[i])
        else:
            walls.append(points[i])

        flag = (flag + 1) % 2

plt.plot([x[1] for x in obst], [x[0] for x in obst], 'ro')
# plt.plot([x[1] for x in points], [x[0] for x in points], 'ro')
plt.show()
