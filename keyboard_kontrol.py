import time

import numpy as np
import threading as thr
from ICP import icp_normal, R, plot_data
from KUKA import YouBot
# from SLAM import SLAM
import matplotlib.pyplot as plt
import pygame as pg
from homogeneous_matrix import *
from PC_methods import *

pg.init()


class RRT_sim:
    def __init__(self, robot=None):

        self.screen = pg.display.set_mode([1000, 1000])
        self.robot = robot
        self.screen_size = 1000
        self.pressed_keys = []
        self.shift = np.array(False)

        self.discrete = 30
        self.robot_radius = int(0.3 * self.discrete + 1)

        self.move_speed_val = 0.5
        self.last_checked_pressed_keys = []
        self.new_map = False
        self.nav_map = np.ones([500, 500])
        self.np_ind = 2
        self.map_shape = self.nav_map.shape
        self.map_k = self.screen_size / max(self.map_shape[0], self.map_shape[1])

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
        move_speed = [0, 0, 0]
        fov = 0
        if pg.K_w in pressed_keys:
            fov += 1
        if pg.K_s in pressed_keys:
            fov -= 1
        move_speed[0] = fov * self.move_speed_val

        rot = 0
        if pg.K_a in pressed_keys:
            rot += 1
        if pg.K_d in pressed_keys:
            rot -= 1
        move_speed[2] = rot * self.move_speed_val

        side = 0
        if pg.K_q in pressed_keys:
            side += 1
        if pg.K_e in pressed_keys:
            side -= 1
        move_speed[1] = side * self.move_speed_val


        if self.last_checked_pressed_keys != pressed_keys:
            self.robot.move_base(*move_speed)
            self.robot.going_to_target_pos = False
            self.last_checked_pressed_keys = pressed_keys[:]

    def main_thr(self):
        while True:
            self.update_keys()
            pg.display.flip()



robot = YouBot('192.168.88.21', ros=True, ssh=False, camera_enable=False, offline=False)#, log=("log/log_complex_wheels.txt", 5))


rrt_sim = RRT_sim(robot)
rrt_sim.main_thr()