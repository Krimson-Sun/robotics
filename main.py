import math
import matplotlib.pyplot as plt


def polar_to_decart(i):
    global coords, lidar, res
    for j in range(len(lidar)):
        angle = coords[i][2] + math.radians(120) - math.radians(0.5 * j)
        x = lidar[i][j] * math.cos(angle)
        y = lidar[i][j] * math.sin(angle)
        ans = ((x + coords[i][0]), y + coords[i][1])
        res.append(ans)


with open("examp12.txt", "r") as f:
    data = f.readlines()

res = []
coords = []
lidar = []

for elem in data:
    tmp = elem.split(';')
    crd = list(map(float, tmp[0].split(', ')))
    ldr = list(map(float, tmp[1].split(', ')))
    coords.append(crd)
    lidar.append(ldr)

for i in range(len(coords)):
    polar_to_decart(i)

plt.plot([x[1] for x in res], [x[0] for x in res], 'ro')
plt.show()
# pygame.init()
# screen = pygame.display.set_mode((1000, 1000))
# surface = pygame.Surface((1000, 1000))
# for x, y in res:
#     surface.set_at((math.floor(100 * x), math.floor(100 * y)), (0, 0, 0))
#     is_running = True
#
# while is_running:
#
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             is_running = False
#     screen.fill((255, 255, 255))
#     screen.blit(surface, (0, 0))
#     pygame.display.update()
# screen.show()


