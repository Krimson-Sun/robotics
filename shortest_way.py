import pygame
from coords_parsing import *
from math import ceil
from rdp import rdp
import numpy as np

pygame.init()
size = width, height = 720, 720
screen = pygame.display.set_mode(size)
scale = 60


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[0] for i in range(height)]
        self.left = 0
        self.top = 0
        self.cell_size = 5
        self.points = self.get_points('examp12.txt')

    def set_view(self, left, top, cell_size):
        self.left = left
        self.top = top
        self.cell_size = cell_size

    def render(self, screen):
        pygame.draw.rect(screen, (153, 153, 255),
                         [self.left, self.top, self.cell_size * self.width, self.cell_size * self.height], 0)
        for point in self.points:
            pygame.draw.rect(screen, (255, 0, 0), [point[0], point[1], 1, 1])

        # for i in range(self.width):
        #     for j in range(self.height):
        #         pygame.draw.rect(screen, pygame.Color(255, 255, 255),
        #                          [self.left + self.cell_size * i,
        #                           self.top + self.cell_size * j,
        #                           self.cell_size, self.cell_size], 1)

    def get_points(self, filename: str):
        res = get_decart_coords(filename)
        res = np.array(res)
        res = rdp(res, epsilon = 0.001, algo='iter')
        print(res[0])


        def convert_coodrs(coords: list):
            global scale
            modified = []
            modifier = tuple(map(abs, [min(x[0] for x in res), min(x[1] for x in res)]))
            for i in range(len(coords)):
                modified.append((ceil((coords[i][0] + modifier[0]) * scale), ceil((coords[i][1] + modifier[1]) * scale)))
            return modified

        points = convert_coodrs(res)
        print(points[0])
        return points


board = Board(200, 200)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        else:
            screen.fill((0, 0, 0))
            board.render(screen)
            pygame.display.flip()



# li = coords_parsing.get_decart_coords("examp12.txt")
