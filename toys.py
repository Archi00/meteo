import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# ------------------------------
# Base Simulation Template
# ------------------------------
class PDEModel(ABC):
    """
    Abstract base class for PDE solvers.
    Defines the template method: setup → step → run.
    """

    def __init__(self, grid_points, dx, dt, total_time):
        self.grid_points = grid_points
        self.dx = dx
        self.dt = dt
        self.total_time = total_time
        self.state = self.initialize_field()

    @abstractmethod
    def initialize_field(self):
        """Define initial condition."""
        pass

    @abstractmethod
    def step(self):
        """Update the field by one timestep."""
        pass

    def run(self):
        """Main loop: advance the PDE in time."""
        snapshots = []
        num_steps = int(self.total_time / self.dt)

        for n in range(num_steps):
            if n % 50 == 0:
                snapshots.append(self.state.copy())
            self.step()

        return snapshots


# ------------------------------
# Concrete Models
# ------------------------------
class HeatEquationSolver(PDEModel):
    """
    Solves the 1D heat equation:
        ∂T/∂t = α ∂²T/∂x²
    using Forward-Time Central-Space (FTCS).
    """

    def __init__(self, grid_points=100, dx=0.01, dt=0.0005, total_time=0.1, diffusivity=0.1):
        self.alpha = diffusivity
        super().__init__(grid_points, dx, dt, total_time)

    def initialize_field(self):
        # Hot spot in the middle
        field = np.zeros(self.grid_points)
        field[self.grid_points // 2] = 100.0
        return field

    def step(self):
        new_state = self.state.copy()
        for i in range(1, self.grid_points - 1):
            new_state[i] = (self.state[i] +
                            self.alpha * self.dt / self.dx**2 *
                            (self.state[i+1] - 2*self.state[i] + self.state[i-1]))
        self.state = new_state


class AdvectionEquationSolver(PDEModel):
    """
    Solves the 1D linear advection equation:
        ∂ϕ/∂t + c ∂ϕ/∂x = 0
    using an upwind scheme.
    """

    def __init__(self, grid_points=100, dx=0.01, dt=0.005, total_time=1.0, velocity=1.0):
        self.c = velocity
        super().__init__(grid_points, dx, dt, total_time)

    def initialize_field(self):
        # Gaussian pulse at the center
        x = np.linspace(0, 1, self.grid_points)
        return np.exp(-200 * (x - 0.5)**2)

    def step(self):
        new_state = self.state.copy()
        for i in range(1, self.grid_points):
            new_state[i] = self.state[i] - self.c * self.dt / self.dx * (self.state[i] - self.state[i-1])
        self.state = new_state


# ------------------------------
# Simulation Runner
# ------------------------------
def visualize_snapshots(snapshots, title):
    plt.figure(figsize=(8, 4))
    for idx, snap in enumerate(snapshots):
        plt.plot(snap, label=f"t={idx}")
    plt.title(title)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Heat equation
    heat_solver = HeatEquationSolver()
    heat_results = heat_solver.run()
    visualize_snapshots(heat_results, "Heat Equation: Diffusion of a Hot Spot")

    # # Advection equation
    # adv_solver = AdvectionEquationSolver()
    # adv_results = adv_solver.run()
    # visualize_snapshots(adv_results, "Advection Equation: Transport of Gaussian Pulse")
