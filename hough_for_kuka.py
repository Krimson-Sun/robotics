import time

import numpy as np
import threading as thr
from ICP import icp_normal, R, find_closest, plot_data
from KUKA import YouBot
# from SLAM import SLAM
import matplotlib.pyplot as plt
import pygame as pg
from homogeneous_matrix import *
from PC_methods import *
import cv2

pg.init()


class RRT_sim:
    def __init__(self, robot=None):

        self.screen = pg.display.set_mode([1000, 1000])
        self.robot = robot
        self.screen_size = 1000
        self.pressed_keys = []
        self.shift = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        self.chunk = np.array(False)

        self.discrete = 30
        self.robot_radius = int(0.3 * self.discrete + 1)

        self.move_speed_val = 0.5
        self.last_checked_pressed_keys = []
        self.new_map = False
        self.nav_map = np.ones([500, 500])
        self.np_ind = 2
        self.map_shape = self.nav_map.shape
        self.map_k = self.screen_size / max(self.map_shape[0], self.map_shape[1])
        self.data = []
        self.record = False
        self.counter = 0
        self.old_lidar = []

    def update_keys(self):

        for event in pg.event.get():
            # Did the user hit a key?
            if event.type == pg.KEYDOWN:
                key = event.key
                if key not in self.pressed_keys:
                    self.pressed_keys.append(key)

                if event.key == pg.K_ESCAPE:
                    running = False
            elif event.type == pg.KEYUP:
                key = event.key
                if key in self.pressed_keys:
                    self.pressed_keys.pop(self.pressed_keys.index(key))
        pressed_keys = self.pressed_keys

        if pg.K_i in pressed_keys:
            print('i')
            np.save(f"poins_test/raw{self.counter}", self.robot.lidar_wheels[1])
            self.counter += 1
            time.sleep(1)
            # self.record = True
        if pg.K_u in pressed_keys:
            print('u')
            self.shift = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
            self.chunk = np.array(False)

        if self.record:
            curr_lid = self.robot.lidar_wheels[1]
            if self.old_lidar != curr_lid:
                self.data.append(curr_lid)
                self.old_lidar = curr_lid
                print(self.counter)
                self.counter += 1
        if pg.K_h in pressed_keys:
            self.counter = 0
            for i in self.data:
                print("iter", self.counter)
                self.counter += 1
                self.new_pc = self.get_lidar(i)
                self.analyse_lidar()
                self.draw_map()
                pg.display.flip()

        if self.last_checked_pressed_keys != pressed_keys:
            self.last_checked_pressed_keys = pressed_keys[:]

    def analyse_lidar(self):
        self.new_pc = self.get_lidar(self.robot.lidar_wheels[1])
        rot = self.shift[:2, :2]
        t = np.array(self.shift[:2, 2])
        compare = np.array(rot.dot(self.new_pc).T + t).T
        if self.chunk.any():
            x, _ = icp_normal(self.new_pc, self.old_pc, 20, 0.9, 0.3)
            self.shift = self.shift @ homogen_matrix_from_pos(x.T[0])
            # plot
            # ax = plot_data(compare, self.chunk, "bef", "log2")
            # rot = R(x[2])
            # t = x[0:2]
            # P_latest = rot.dot(compare) + t
            # ax = plot_data(self.chunk, P_latest, "raw", "log2")
            # plt.show()
        else:
            self.chunk = self.new_pc

        rot = self.shift[:2, :2]
        t = np.array(self.shift[:2, 2])
        P_latest = np.array(rot.dot(self.new_pc).T + t).T

        self.old_pc = self.new_pc.copy()
        # self.imprint(P_latest)

        add_to_chunk = []
        for i in range(len(P_latest[0])):
            ind, dist = find_closest(P_latest[:, i], self.chunk)
            if dist > 0.05:
                add_to_chunk.append(i)

        if len(add_to_chunk) != 0:
            self.chunk = np.append(self.chunk, P_latest[:, add_to_chunk], axis=1)
        # else:
        #     print(-1)

        order = np.zeros(len(self.chunk[1]))
        curr_obj = 1
        points_checked = 1
        p_i = list(range(1, len(self.chunk[0])))
        rest = self.chunk[:, 1:]
        curr_point = self.chunk[:, 0]
        while True:
            if points_checked >= len(self.chunk[0]):
                break

            # if curr_point_ind == 0:
            #     buff_rest = rest[:, 1:]
            # else:
            #     buff_rest = np.concatenate((rest[:, :curr_point_ind], rest[:, curr_point_ind + 1:]), axis=1)

            ind, dist = find_closest(curr_point, rest)
            if ind ==-1:
                print(curr_point, rest)
                break

            curr_point = rest[:, ind]

            cl_orig_ind = int(p_i[ind])

            if ind == 0:
                rest = rest[:, 1:]
                p_i = p_i[1:]
            else:
                rest = np.concatenate((rest[:, :ind], rest[:, ind + 1:]), axis=1)
                p_i = np.concatenate((p_i[:ind], p_i[ind + 1:]), axis=0)
            order[cl_orig_ind] = curr_obj
            curr_obj += 1
            points_checked += 1

        self.chunk = self.chunk[:, np.argsort(order)]
        self.nav_map = np.ones_like(self.nav_map)
        self.imprint(self.chunk)

    def imprint(self, points):
        for i in points.T:
            try:
                x, y = i * self.discrete + np.array(self.map_shape) / 2
                self.nav_map[int(x), int(y)] = 0
            except:
                pass

    def get_points_from_map(self):
        print((np.array(np.where(self.nav_map == 0)).T - np.array(self.map_shape) / 2) / self.discrete)

    def draw_map(self):
        if self.nav_map.any():
            map_img = pg.transform.scale(pg.surfarray.make_surface((self.nav_map * -1 + 1) * 255),
                                         (self.map_shape[0] * self.map_k, self.map_shape[1] * self.map_k))
            self.screen.blit(map_img, (0, 0))

    def draw_curr_pos(self):
        if self.robot.increment_by_wheels:
            half = self.map_shape[0] // 2
            pg.draw.circle(self.screen, (0, 0, 102), list(
                map(lambda x: (half + x * self.discrete) * self.map_k, self.robot.increment_by_wheels[:2])), 10)

    def main_thr(self):
        running = True
        while running:
            # inc = self.robot.lidar[0]
            # rot = R([inc[2]])
            # t = inc[:2]
            # P_latest = rot.dot((from_radial_to_cartesian(np.array(self.robot.lidar_wheels[1]))[::10]).T).T + t
            # self.imprint(P_latest.T)
            # print(P_latest)
            # print(self.robot.odom_speed)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    print(-1)
                    self.get_walls()
            self.update_keys()
            self.draw_map()
            pg.display.flip()


    def lidar_cycle(self):
        while not self.robot.lidar_wheels[1]:
            print("waiting")
            time.sleep(0.8)
        self.old_pc = self.get_lidar(self.robot.lidar_wheels[1])
        print("start")
        while thr.main_thread().is_alive():
            self.analyse_lidar()
            time.sleep(0.05)

    def get_lidar(self, d):
        curr_lidar = from_radial_to_cartesian(np.array(d))
        out_ind = smooth(curr_lidar, step=0.0001, threshold=0.1)
        return curr_lidar[out_ind].T

    def get_walls(self):
        image = np.array(self.nav_map)
        np.place(image, image<1, [255])
        np.place(image, image<240, [0])

        # for i in range(len(image)):
        #     for j in range(len(image[i])):
        #         if image[i][j] == 0:
        #             image[i][j] == 255
        #         else:
        #             image[i][j] = 0
        image8 = np.asarray(image, dtype=np.uint8)
        for elem in image:
            print(elem)
        cv2.imshow("Probabilistic Line Transform", image8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        lines = cv2.HoughLinesP(image8, 1, np.pi/360, 1, 2, 10)
        for i in range(len(lines)):
            l = lines[i][0]
            cv2.line(image8, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA)
        cv2.imshow("Probabilistic Line Transform", image8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


robot = YouBot('192.168.88.21', ros=False, ssh=False, camera_enable=False, offline=False)  # , log=("log/log6.txt", 5))

rrt_sim = RRT_sim(robot)
lidar_thr = thr.Thread(target=rrt_sim.lidar_cycle)
lidar_thr.start()
rrt_sim.main_thr()

