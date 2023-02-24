import pygame
import coords_parsing

pygame.init()
size = width, height = 500, 700
screen = pygame.display.set_mode()

class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.board = [[0] for i in range(height)]
        self.left = 12
        self.top = 12
        self.cell_size = 30

    def set_view(self, left, top, cell_size):
        self.left =left
        self.top = top
        self.cell_size = cell_size

    def render(self, screen):
        pygame.draw.rect(screen, (153, 153, 255),
                         [self.left, self.top, self.cell_size * 12, self.cell_size * 12], 0)
        for i in range(self.width):
            for j in range(self.height):
                pygame.draw.rect(screen, pygame.Color(255, 255, 255),
                                 [self.left + self.cell_size * i,
                                  self.top + self.cell_size * j,
                                  self.cell_size, self.cell_size], 1)

board = Board(5, 7)
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

