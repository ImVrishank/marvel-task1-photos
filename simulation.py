import pygame
import torch

# -------------------- CONFIG --------------------
WIDTH, HEIGHT = 600, 400
NUM_PARTICLES = 100
PARTICLE_RADIUS = 5
FPS = 60
DT = 1 / 30
GRAVITY = torch.tensor([0.0, 980.0])  # px/s^2 downward
MAX_SPEED = 150.0

# -------------------- INIT --------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fluid Simulation")
clock = pygame.time.Clock()

device = torch.device("cpu")

# Particle states
positions = torch.rand((NUM_PARTICLES, 2), device=device)
positions[:, 0] *= WIDTH - 2 * PARTICLE_RADIUS
positions[:, 0] += PARTICLE_RADIUS
positions[:, 1] *= HEIGHT - 2 * PARTICLE_RADIUS
positions[:, 1] += PARTICLE_RADIUS

velocities = torch.zeros((NUM_PARTICLES, 2), device=device)

# -------------------- SIMULATION --------------------
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # ---------- PHYSICS ----------

    # Random movement (placeholder for PINN)
    random_force = (torch.rand_like(velocities) - 0.5) * 2 * MAX_SPEED
    velocities += random_force * DT

    # Gravity
    velocities += GRAVITY * DT

    # Update positions
    positions += velocities * DT

    # Wall collisions
    # X-axis
    mask_left = positions[:, 0] < PARTICLE_RADIUS
    mask_right = positions[:, 0] > WIDTH - PARTICLE_RADIUS
    positions[mask_left, 0] = PARTICLE_RADIUS
    positions[mask_right, 0] = WIDTH - PARTICLE_RADIUS
    velocities[mask_left | mask_right, 0] *= -0.5

    # Y-axis
    mask_top = positions[:, 1] < PARTICLE_RADIUS
    mask_bottom = positions[:, 1] > HEIGHT - PARTICLE_RADIUS
    positions[mask_top, 1] = PARTICLE_RADIUS
    positions[mask_bottom, 1] = HEIGHT - PARTICLE_RADIUS
    velocities[mask_top | mask_bottom, 1] *= -0.5

    # Particle-particle collisions (brute force)
    for i in range(NUM_PARTICLES):
        for j in range(i + 1, NUM_PARTICLES):
            delta = positions[j] - positions[i]
            dist = torch.norm(delta)
            min_dist = 2 * PARTICLE_RADIUS
            if dist < min_dist and dist > 0:
                n = delta / dist
                overlap = min_dist - dist
                positions[i] -= 0.5 * overlap * n
                positions[j] += 0.5 * overlap * n

                # simple elastic response along normal
                vi_n = torch.dot(velocities[i], n)
                vj_n = torch.dot(velocities[j], n)
                velocities[i] += (vj_n - vi_n) * n
                velocities[j] += (vi_n - vj_n) * n

    # ---------- RENDER ----------
    screen.fill((20, 20, 30))
    for pos in positions:
        pygame.draw.circle(screen, (0, 200, 255), (int(pos[0]), int(pos[1])), PARTICLE_RADIUS)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
