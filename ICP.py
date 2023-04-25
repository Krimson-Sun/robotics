import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from math import sin, cos, atan2, pi
# from scipy import spatial
from IPython.display import display, Math, Latex, Markdown, HTML
from numba import njit
from PC_methods import *


### plotting ###
def plot_data(data_1, data_2, label_1, label_2, markersize_1=8, markersize_2=8):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.axis('equal')
    if data_1 is not None:
        x_p, y_p = data_1
        ax.plot(x_p, y_p, color='#336699', markersize=markersize_1, marker='o', linestyle=":", label=label_1)
    if data_2 is not None:
        x_q, y_q = data_2
        ax.plot(x_q, y_q, color='orangered', markersize=markersize_2, marker='o', linestyle=":", label=label_2)
    ax.legend()
    return ax


def plot_values(values, label):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(values, label=label)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_normals(normals, ax):
    label_added = False
    for normal in normals:
        if not label_added:
            ax.plot(normal[:, 0], normal[:, 1], color='grey', label='normals')
            label_added = True
        else:
            ax.plot(normal[:, 0], normal[:, 1], color='grey')
    ax.legend()
    return ax


### end plotting ###


### ICP math ###
@njit(fastmath=True)
def dR(theta):
    theta = theta
    return np.array([[-sin(theta), -cos(theta)],
                     [cos(theta), -sin(theta)]])

@njit(fastmath=True)
def R(theta):
    theta = theta[0]
    return np.array([[cos(theta), -sin(theta)],
                     [sin(theta), cos(theta)]])

@njit(fastmath=True)
def jacobian(x, p_point):
    theta = x[2]
    J = np.zeros((2, 3))
    J[0:2, 0:2] = np.identity(2)
    J[0:2, 2] = dR(theta[0]).dot(p_point).T
    return J

@njit(fastmath=True)
def error(x, p_point, q_point):
    rotation = R(x[2])
    translation = x[0:2].T
    prediction = rotation.dot(p_point) + translation
    return (prediction - q_point)[0]

@njit(fastmath=True)
def compute_normals(points):
    normals = np.zeros((points.shape[1], 2))
    for i in range(1, len(points[0]) - 1):
        prev_point = points[:, i - 1]
        next_point = points[:, i + 1]
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        normal = np.array([-dy, dx])
        normals[i] = normal / np.linalg.norm(normal)
    normals[-1, :] = np.array([0, 0])
    return normals

@njit(fastmath=True)
def find_closest(point, other):
    min_dist = -1
    min_ind = -1
    for i in range(0, len(other[0])):
        op = other[:, i]
        dist = np.linalg.norm(op - point)
        if dist < min_dist or min_dist == -1:
            min_dist = dist
            min_ind = i
    return min_ind, min_dist

@njit(fastmath=True)
def prepare_data(points1, points2, normals1, normals2):
    # A_tree = spatial.KDTree(points2)
    # dist, indexes = A_tree.query(points1)
    # dst_i = indexes.ravel()
    cl_i, cl_dst = find_closest(points1[:, 0], points2)
    corresp = np.zeros((points1.shape[1], 2)).astype(np.int32)
    corresp_dist = np.zeros(points1.shape[1])
    corresp[0] = np.array([[0, cl_i]])
    corresp_dist[0] = cl_dst

    last = 0
    curr = 1
    for i in range(1, len(points1[0])):
        last = curr
        cl_i, cl_dst = find_closest(points1[:, i], points2)
        if cl_i == corresp[curr-1][1]:
            if corresp_dist[curr-1] >= cl_dst:
                corresp_dist[curr-1] = cl_dst
        else:
            corresp[curr] = np.array([i, cl_i])
            corresp_dist[curr] = np.array(cl_dst)
            curr += 1
    corresp = corresp[:last]
    normals = np.zeros((last, 2))
    for num in range(last):
        i, j = corresp[num]

        curr_norm = normals1[i] + normals2[j]
        if np.linalg.norm(curr_norm) != 0:
            normals[num] = (curr_norm / np.linalg.norm(curr_norm))
        else:
            normals[num] = np.array([0.0,0.0])

    # ax = plot_data(src.T, dst.T, "lbl", "lbl2")
    # plot_normals(np.stack((src, normals+src), axis = 1), ax)
    # plt.show()
    return normals, corresp, corresp_dist

@njit(fastmath=True)
def prepare_system_normals(x, P, Q, normals, corresp, thr):
    H = np.zeros((3, 3))
    g = np.zeros((3, 1))
    chi = 0
    for num in range(len(corresp)):
        i, j = corresp[num]
        p_point = P[:, int(i)]
        q_point = Q[:, int(j)]
        normal = normals[num]
        e = normal.dot(error(x, p_point, q_point))
        J = np.array([[*normal.dot(jacobian(x, p_point))]])
        weight = kernel(thr, e)
        H += weight * J.T.dot(J)
        g += weight * J.T * e
        chi += e
    return H, g, chi

@njit(fastmath=True)
def icp_normal(P, Q, iterations=30, thr_k=0.8, init_thr=100):
    norm1_init = compute_normals(P)
    norm2 = compute_normals(Q)

    x = np.zeros((3, 1))
    thr = init_thr
    chi_values = []
    x_values = [x.copy()]  # Initial value for transformation.
    P_values = [P.copy()]
    P_latest = P.copy()
    # corresp_values = []
    for i in range(iterations):
        rot = R(x[2])
        t = x[0:2]
        norm1 = rot.dot(norm1_init.T).T
        norm, corresp, _ = prepare_data(P_latest, Q, norm1, norm2)
        H, g, chi = prepare_system_normals(x, P, Q, norm, corresp, thr)
        thr *= thr_k
        dx = np.linalg.lstsq(H, -g, rcond=-1)[0]
        x += dx
        x[2] = atan2(sin(x[2][0]), cos(x[2][0]))  # normalize angle
        chi_values.append(chi)  # add error to list of errors
        x_values.append(x.copy())
        rot = R(x[2])
        t = x[0:2]
        P_latest = rot.dot(P.copy()) + t
        P_values.append(P_latest)
    # corresp_values.append(corresp_values[-1])
    return x, chi*-1#, corresp



@njit(fastmath=True)
def kernel(threshold, error):
    #return 1.0 / (error ** 4 + 0.3)
    if error < threshold:
        return 1.0 # / (error ** 4 + 0.3)
    return 0.0
#print(threshold / (error ** 4 + threshold))


# initialize pertrubation rotation
if __name__ == "__main__":

    points1 = from_radial_to_cartesian(np.load("poins_test/raw0.npy"))
    #points1 = points1[smooth(points1, step=0.01, threshold=0.02)]
    points2 = from_radial_to_cartesian(np.load("poins_test/raw1.npy"))
    #points2 = points2[smooth(points2, step=0.01, threshold=0.02)]
    angle = pi / 4
    R_true = np.array([[cos(angle), -sin(angle)],
                       [sin(angle), cos(angle)]])
    t_true = np.array([[-2], [5]])

    # Generate data as a list of 2d points
    num_points = 30
    true_data = np.zeros((2, num_points))
    true_data[0, :] = range(0, num_points)
    true_data[1, :] = 0.2 * true_data[0, :] * np.sin(0.5 * true_data[0, :])
    # Move the data
    moved_data = R_true.dot(true_data) + t_true

    # Assign to variables we use in formulas.
    Q = true_data
    P = moved_data

    #plot_data(moved_data, true_data, "P: moved data", "Q: true data")
    #plt.show()
    # points2 = Q.T
    # points1 = P.T
    print(points1.shape, points2.shape)
    x, _, = icp_normal(points1.T, points2.T)
    print(_)
    rot = R(x[2])
    t = x[0:2]
    P_latest = rot.dot(points1.T) + t


    ax = plot_data(points2.T, P_latest, "log1", "log2new")
    #ax = plot_data(points2.T, points1.T, "log1", "log2")
    plt.show()