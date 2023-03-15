import pygame
import glob
import time
from PIL import Image


def pilImageToSurface(pil_image):
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
        ).convert()


pygame.init()
window = pygame.display.set_mode((1024, 1024))
clock = pygame.time.Clock()

pil_images = [Image.open(path) for path in glob.glob('out/*.png')]

run = True
last_time = time.time()
index = 0
current_surface = pilImageToSurface(pil_images[index])
interval = 1.0
while run:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    current_time = time.time()
    if current_time > last_time + interval:
        index += 1
        if index >= len(pil_images): index = 0
        current_surface = pilImageToSurface(pil_images[index])
        last_time = current_time
        print(index)

    # window.fill(0)
    window.blit(current_surface, current_surface.get_rect(center=(512, 512)))
    pygame.display.flip()