import torch
import pygame
import numpy as np

# ---------------- Config ----------------
DEVICE = "cuda"  # run on GPU
WIDTH, HEIGHT = 600, 400
NUM_PARTICLES = 200
RADIUS = 5
MASS = 1.0
DT = 0.01
GRAVITY = torch.tensor([0, 981.0], device=DEVICE)  # pixels/s²
MAX_SPEED = 500.0

# ---------------- Init Particles ----------------
pos = torch.rand(NUM_PARTICLES, 2, device=DEVICE)
pos[:, 0] *= WIDTH - 2*RADIUS
pos[:, 0] += RADIUS
pos[:, 1] *= HEIGHT - 2*RADIUS
pos[:, 1] += RADIUS

vel = torch.zeros_like(pos, device=DEVICE)

# ---------------- Pygame Setup ----------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CUDA Particle Fluid")
clock = pygame.time.Clock()
FPS = 60

# ---------------- Simulation Loop ----------------
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Apply gravity
    vel += GRAVITY * DT

    # Update positions
    pos += vel * DT

    # ---------------- Particle-Wall Collisions ----------------
    # X walls
    mask = pos[:, 0] < RADIUS
    pos[mask, 0] = RADIUS
    vel[mask, 0] *= -0.8
    mask = pos[:, 0] > WIDTH - RADIUS
    pos[mask, 0] = WIDTH - RADIUS
    vel[mask, 0] *= -0.8

    # Y walls
    mask = pos[:, 1] < RADIUS
    pos[mask, 1] = RADIUS
    vel[mask, 1] *= -0.8
    mask = pos[:, 1] > HEIGHT - RADIUS
    pos[mask, 1] = HEIGHT - RADIUS
    vel[mask, 1] *= -0.8

    # ---------------- Particle-Particle Collisions ----------------
    # naive O(N^2) approach (good for <= 500 particles on GPU)
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)  # NxN x/y differences
    dist = torch.norm(delta, dim=2) + 1e-6
    overlap = 2*RADIUS - dist
    coll_mask = (overlap > 0) & (~torch.eye(NUM_PARTICLES, dtype=bool, device=DEVICE))

    if coll_mask.any():
        # resolve collisions by simple impulse
        impulse = 0.5 * overlap[coll_mask].unsqueeze(1) * delta[coll_mask] / dist[coll_mask].unsqueeze(1)
        idxs = torch.nonzero(coll_mask)
        pos[idxs[:,0]] += impulse
        pos[idxs[:,1]] -= impulse
        vel[idxs[:,0]] -= impulse
        vel[idxs[:,1]] += impulse

    # ---------------- Draw ----------------
    screen.fill((20, 20, 30))
    positions_cpu = pos.cpu().numpy()
    for p in positions_cpu:
        pygame.draw.circle(screen, (0, 200, 255), (int(p[0]), int(p[1])), RADIUS)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
