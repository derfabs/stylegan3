from typing import Tuple

import math
import click
import multiprocessing as mp
from multiprocessing.connection import Connection
import pygame
import time
import torch
import numpy as np
import dnnlib
import legacy
from PIL import Image as image
from PIL.Image import Image


def parse_dimensions(input: str) -> Tuple[str]:
    return tuple(int(dim.strip()) for dim in input.split(','))


def generate_frames(network: str, speed: float, seed: float, pipe: Connection):
    # check for cuda
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print('cuda is available.')
        device = torch.device('cuda')
    else:
        print('cuda is not available.')
        device = torch.device('cpu')
    print(f'device: "{device}"')

    # load model
    print(f'Loading networks from "{network}"...')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # generate label
    label = torch.zeros([1, G.c_dim], device=device)

    speed = 0.2  # added to the seed every second

    last_time = time.time()
    seed_lower = None
    while True:
        # get time past since last iteration
        interval = time.time() - last_time
        last_time = time.time()
        seed += interval * speed

        # generate latent vectors
        if seed_lower != math.floor(seed):
            seed_lower = int(math.floor(seed))
            z_lower = torch.from_numpy(
                np.random.RandomState(seed_lower).randn(1, G.z_dim)
                ).to(device)

            seed_upper = int(math.ceil(seed))
            if seed_upper == seed_lower: seed_upper = seed_lower + 1
            z_upper = torch.from_numpy(
                np.random.RandomState(seed_upper).randn(1, G.z_dim)
                ).to(device)

        t = seed - seed_lower
        z = z_lower * (1 - t) + z_upper * t

        # generate image
        img: torch.Tensor = G(z, label, truncation_psi=1, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5
               + 128).clamp(0, 255).to(torch.uint8)

        pil_image = image.fromarray(img[0].cpu().numpy(), 'RGB')

        pipe.send(pil_image)


def pilImageToSurface(pil_image: Image) -> pygame.Surface:
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
        ).convert()


# yapf: disable
@click.command()
@click.option('--network',            help='model network pickle filename', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option('--window_dimensions',  help='width & height of the window, comma seperated', type=parse_dimensions, required=True)
@click.option('--speed',              help='base speed', type=float, required=True)
@click.option('--fps',                help='how many frames/sec to render', type=int, default=60, show_default=True)
@click.option('--seed',               help='starting seed', type=float, default=0.0, show_default=True)
# yapf: enable
def stream(
        network: str,
        window_dimensions: Tuple[int, int],
        speed: float,
        fps: int,
        seed: float
    ) -> None:

    # setup pygame
    pygame.init()
    window = pygame.display.set_mode(window_dimensions)
    clock = pygame.time.Clock()

    # setup pipe
    parent_conn, child_conn = mp.Pipe()

    image_dimensions = (min(window_dimensions), min(window_dimensions))
    # start process
    process = mp.Process(
        target=generate_frames,
        kwargs=({
            'network': network,
            'speed': speed,
            'seed': seed,
            'pipe': child_conn
            })
        )

    process.start()

    current_surface: pygame.Surface = None
    try:
        while True:
            # limit fps
            clock.tick(fps)

            # quite the loop when closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()

            # get new surface if avaliable
            if parent_conn.poll():
                pil_image: Image = parent_conn.recv()

                if pil_image.size != image_dimensions:
                    pil_image = pil_image.resize(image_dimensions)

                current_surface = pilImageToSurface(pil_image)

            # draw current surface if avaliable
            if current_surface:
                window.fill(0)
                window.blit(
                    current_surface,
                    current_surface.get_rect(
                        center=tuple(el / 2 for el in window_dimensions)
                        )
                    )
            else:
                window.fill(0)

            # update window
            pygame.display.flip()
    except:
        if process: process.terminate()
        if process: process.join()
        if parent_conn: parent_conn.close()
        if child_conn: child_conn.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    stream()