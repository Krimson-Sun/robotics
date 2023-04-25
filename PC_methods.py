import numpy as np
import matplotlib.pyplot as plt
from homogeneous_matrix import *


def from_radial_to_cartesian(points):
    lidar = points
    xy_local_form = []
    for i in range(100, len(lidar)-100):
        if lidar[i] > 5.5 or lidar[i - 1] > 5.5 or lidar[i] < 0.5 or lidar[i - 1] < 0.5:
            continue
        if abs(lidar[i - 1] - lidar[i]) < 0.1:
            lid_ang = i * math.radians(240) / 681 - math.radians(30)
            lid_dist = lidar[i]
            if lid_dist > 5.5:
                continue
            ox = lid_dist * math.sin(lid_ang)
            oy = lid_dist * math.cos(lid_ang)
            xy_local_form.append((ox, oy))
    xy_local_form = np.array(xy_local_form)

    return np.array(xy_local_form)


def xy_local_to_global(xy_local_form, odom):
    out_points = np.array(xy_local_form)
    converted = np.ones((out_points.shape[1] + 1, out_points.shape[0]))
    converted[:out_points.shape[1], :] = np.copy(out_points.T)
    # transform
    converted = np.dot(homogen_matrix_from_pos(odom, True), converted)
    # back from homogeneous to cartesian
    converted = np.array(converted[:converted.shape[1], :]).T
    return np.array(converted[:, :2])


def find_farthest(points, threshold=1.0):
    x1, y1 = points[0]
    x2, y2 = points[-1]
    main_vert = False
    perp_vert = False
    a_main = 0
    a_perp = 0
    max_dist = -1
    max_i = None
    if x1 == x2:
        main_vert = True
    else:
        a_main = (y1 - y2) / (x1 - x2)
    b_main = y1 - a_main * x1
    if a_main != 0:
        a_perp = -1 / a_main
    else:
        perp_vert = True
    for i in range(1, len(points) - 2):
        xp, yp = points[i]
        if main_vert:
            dist = xp ** 2
        elif perp_vert:
            dist = yp ** 2
        else:
            b_perp = yp - a_perp * xp
            xc = (b_main - b_perp) / (a_perp - a_main)
            yc = xc * a_main + b_main
            dist = (xp - xc) ** 2 + (yp - yc) ** 2
        if dist > max_dist:
            max_dist = dist
            max_i = i
    if max_dist < threshold:
        return None
    if max_dist == -1:
        return None
    return max_i


def binary_search(arr, x):
    low, high = 0, len(arr) - 1
    mid = 0
    mid_old = 0
    while high >= low:
        mid = (high + low) // 2
        if arr[mid] == x:
            return None
        elif arr[mid] > x:
            high = mid
        else:
            low = mid
        if mid_old == high + low:
            break
        mid_old = high + low
    return mid + 1


def douglas_peucker(points, /, iters=30, min_weight=0.1, threshold=0.03):
    all_points = len(points)
    point_list = [0, all_points - 1]
    weights = [0]
    new_point_list = [0, all_points - 1]

    for a in range(iters):
        for i in range(1, len(point_list)):
            add_point = None
            if point_list[i] - point_list[i - 1] > 5:
                add_point = find_farthest(points[point_list[i - 1]:point_list[i]], threshold)
            if not add_point:
                weights[i - 1] += point_list[i] - point_list[i - 1]
                continue
            add_point += point_list[i - 1]
            address = binary_search(new_point_list, add_point)
            if not address:
                continue
            weights.insert(address, 0)
            new_point_list.insert(address, add_point)
        if point_list == new_point_list:
            break
        point_list = new_point_list[:]
    out_points = []
    f = 0
    while f < len(weights) - 1 and weights[f] < min_weight:
        f += 1
    out_points.append(point_list[f])
    for i in range(len(weights)):
        if weights[i] > min_weight:
            if not out_points:
                out_points.append(point_list[i])
            out_points.append(point_list[i + 1])
    return np.array(out_points), weights


def smooth(points, /, step, threshold):
    corners, _ = douglas_peucker(points, threshold=threshold, iters=int(25 / step))  # iters = max_len/step
    curr_corner = 1
    out_ind = [corners[0]]
    curr_dist = 0
    for i in range(1, len(points)):
        if i >= corners[curr_corner]:
            out_ind.append(corners[curr_corner])
            curr_corner += 1
        curr_dist += np.linalg.norm(points[i] - points[i - 1])
        if curr_dist > step:
            out_ind.append(i)
            curr_dist = 0
    return out_ind


if __name__ == "__main__":
    points1 = np.load("poins_test/points2.npy")
    out_ind = smooth(points1, step=0.5, threshold=0.2)
    points1 = points1[out_ind]
    plt.plot(points1.T[0], points1.T[1], marker='o')
    plt.axis('equal')
    plt.show()