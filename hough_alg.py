import math
import scipy
import numpy as np
import cv2
import matplotlib.pyplot as plt

li = np.load('coordinates/coords2.npy')

map_size = max((max(elem[1] for elem in li) - min(elem[0] for elem in li)),
               (max(elem[0] for elem in li) - min(elem[0] for elem in li)))
def hough_transform_dec(input, map_size):
    resolution = int(map_size * 2.5)
    approx_reg = int(map_size * 0.2)
    min_detect_val = 0.08
    approx_reg_mid = int(approx_reg / 2)
    max_point_val = len(input)
    detected_peaks_params = []

    hough_graph = np.zeros((resolution, resolution))
    for i in range(len(input)):
        for j in range(1, resolution):
            point = int(input[i][1] * math.cos(math.pi / resolution * j) - input[i][0] * math.sin(
                math.pi / resolution * j) + resolution / 2)
            if abs(point) < resolution:
                hough_graph[point][j] += 1 / max_point_val

    resized = cv2.resize(hough_graph, (500, 500), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("hough_graph.jpg", resized)


     # define an 8-connected neighborhood
    n_mask = scipy.ndimage.generate_binary_structure(2, 1)
    neighborhood = np.zeros((approx_reg, approx_reg))
    neighborhood[approx_reg_mid][approx_reg_mid] = 1
    neighborhood = scipy.ndimage.binary_dilation(neighborhood, structure=n_mask).astype(n_mask.dtype)
    for i in range(int(approx_reg_mid / 3)):
        neighborhood = scipy.ndimage.binary_dilation(neighborhood, structure=neighborhood).astype(n_mask.dtype)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = scipy.ndimage.maximum_filter(hough_graph, size=20) == hough_graph

    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (hough_graph < min_detect_val)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = scipy.ndimage.binary_erosion(background, structure=neighborhood, border_value=1)
    background_inv = (eroded_background == False)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (and operation)
    detected_peaks = local_max & background_inv

    for i in range(len(detected_peaks)):
        for j in range(len(detected_peaks)):
            if detected_peaks[i][j]:
                f, p, = math.pi / resolution * j, i - resolution / 2
                detected_peaks_params.append([f, p, hough_graph[i][j]])
    detected_peaks_params = sorted(detected_peaks_params, key=lambda x: x[-1], reverse=True)
    return detected_peaks_params


def get_walls(points):
    map_size = max((max(elem[1] for elem in points) - min(elem[0] for elem in points)),
                   (max(elem[0] for elem in points) - min(elem[0] for elem in points)))
    detected_peaks = hough_transform_dec(points, map_size)
    line_params = []
    for elem in detected_peaks:
        if elem[2] > 0.0015:
            line_params.append([elem[0], elem[1]])
    lines = []
    for angle, param in line_params:
        if ((param / math.cos(angle)) <= map_size) and ((param / math.cos(angle)) > 0):
            lines.append([0, param / math.cos(angle)])
        elif (-param / math.sin(angle) > 0) and (-param / math.sin(angle) < map_size):
            lines.append([-param / math.sin(angle), 0])
        elif ((map_size * math.sin(angle) + param) / math.cos(angle)) > 0 and ((map_size * math.sin(angle)
                                                                                + param) / math.cos(angle)):
            lines.append([map_size, (map_size * math.sin(angle) + param) / math.cos(angle)])
    return lines

detected_peaks = hough_transform_dec(li, map_size)
print(detected_peaks)
line_params = []
for elem in detected_peaks:
    line_params.append([elem[0], elem[1]])
lines = []
for angle, param in line_params:
    if ((param / math.cos(angle)) <= map_size) and ((param / math.cos(angle)) > 0):
        lines.append([angle, param])
    elif (-param / math.sin(angle) > 0) and (-param / math.sin(angle) < map_size):
        lines.append([angle, param])
    elif ((map_size * math.sin(angle) + param) / math.cos(angle)) > 0 and ((map_size * math.sin(angle)
                                                                            + param) / math.cos(angle)):
        lines.append([angle, param])
print(lines)

def draw_lines(angle, param):
    x = np.linspace(0, 1250, 1250)
    y = (x * np.sin(angle) + param) / np.cos(angle)
    plt.plot(x, y, '-r')

for line in lines:
    draw_lines(line[0], line[1])
plt.show()

# print(len(lines))

# line_params = [[elem[0], elem[1]] for elem in detected_peaks_params]




