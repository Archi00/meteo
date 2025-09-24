import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

nx, ny = 200, 120
Lx, Ly = 1.0, 0.6
dx, dy = Lx/(nx-1), Ly/(ny-1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# initial Gaussian blob
sigma = 0.06
u0 = np.exp(-((X-0.3)**2 + (Y-0.3)**2)/(2*sigma**2))

# constant wind field (m/nondim units)
vx, vy = 0.3, 0.05

# CFL-stable dt
dt = 0.4 * min(dx/abs(vx) if vx!=0 else 1e6, dy/abs(vy) if vy!=0 else 1e6)

u = u0.copy()

def advect(u, vx, vy, dx, dy, dt):
    un = u.copy()
    # x-direction upwind
    if vx >= 0:
        ux = (un - np.roll(un, 1, axis=1)) / dx
    else:
        ux = (np.roll(un, -1, axis=1) - un) / dx
    # y-direction upwind
    if vy >= 0:
        uy = (un - np.roll(un, 1, axis=0)) / dy
    else:
        uy = (np.roll(un, -1, axis=0) - un) / dy
    return un - dt*(vx * ux + vy * uy)

fig, ax = plt.subplots(figsize=(8,4))
im = ax.imshow(u, origin='lower', extent=[0,Lx,0,Ly], vmin=0, vmax=1)
ax.set_title('2D advection (upwind)')
plt.colorbar(im, ax=ax)

def update(i):
    global u
    u = advect(u, vx, vy, dx, dy, dt)
    # add simple boundary damping to avoid wrap-around if you like:
    im.set_data(u)
    ax.set_xlabel(f"step {i}")
    return [im]

ani = FuncAnimation(fig, update, frames=400, interval=30)
plt.show()
