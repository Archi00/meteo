# heat1d.py (or Jupyter cell)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain (nondimensional toy model)
nx = 201
L = 1.0
dx = L/(nx-1)
x = np.linspace(0, L, nx)

kappa = 1e-3         # diffusion coefficient (toy)
dt = 0.25 * dx**2 / kappa   # safe dt
nt = 800

# initial: gaussian bump in middle
u = np.exp(-((x-0.5)**2)/(2*(0.08)**2))

fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_ylim(0, 1.1)
ax.set_title("1D heat equation (toy)")

def step(u):
    un = u.copy()
    # interior points
    u[1:-1] = un[1:-1] + kappa * dt / dx**2 * (un[2:] - 2*un[1:-1] + un[:-2])
    # Neumann BC (zero gradient)
    u[0] = u[1]
    u[-1] = u[-2]
    return u

def update(i):
    global u
    # optional diurnal forcing: add small sinusoidal forcing at center:
    forcing = 0.0
    if (i % 200) < 100:
        forcing = 0.001 * np.exp(-((x-0.5)**2)/(2*(0.03)**2))
    u = step(u) + forcing
    line.set_ydata(u)
    ax.set_xlabel(f"step {i}, max={u.max():.3f}")
    return line,

ani = FuncAnimation(fig, update, frames=nt, blit=True, interval=30)
plt.show()
