import math
import matplotlib.pyplot as plt
import numpy as np

scale = 60

def polar_to_decart(i, coords, lidar, res):
    for j in range(len(lidar)):
        if lidar[i][j] == 5.6 or lidar[i][j] < 0.3:
            continue
        else:
            angle = coords[i][2] + math.radians(120) - math.radians(0.5 * j)
            x = lidar[i][j] * math.cos(angle)
            y = lidar[i][j] * math.sin(angle)
            ans = ((x + coords[i][0]), y + coords[i][1])
            res.append(ans)

def coords_pars(data, crds, ldar):
    for elem in data:
        tmp = elem.split(';')
        crd = list(map(float, tmp[0].split(', ')))
        ldr = list(map(float, tmp[1].split(', ')))
        crds.append(crd)
        ldar.append(ldr)


def get_decart_coords(FileName):
    with open(FileName, "r") as f:
        data = f.readlines()

    res = []
    coords = []
    lidar = []

    coords_pars(data, coords, lidar)

    for i in range(len(coords)):
        polar_to_decart(i, coords, lidar, res)

    return res

def get_points(filename: str):
    res = get_decart_coords(filename)
    res = np.array(res)
    # res = rdp(res, epsilon=0.01, algo='iter')


    def convert_coodrs(coords: list):
        global scale
        modified = []
        modifier = tuple(map(abs, [min(x[0] for x in res), min(x[1] for x in res)]))
        for i in range(len(coords)):
            modified.append((math.ceil((coords[i][0] + modifier[0]) * scale), math.ceil((coords[i][1] + modifier[1]) * scale)))
        return modified

    points = convert_coodrs(res)
    return points

li = np.array(get_points('coordinates/examp2.txt'))

np.save('coords2', li)

# res = get_decart_coords()
# print(len(res))
# print(min(x[0] for x in res), max(x[0] for x in res))
# print(min(x[1] for x in res), max(x[1] for x in res))
# plt.plot([x[1] for x in res], [x[0] for x in res], 'ro')
# plt.show()
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


