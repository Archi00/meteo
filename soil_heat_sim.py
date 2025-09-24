import numpy as np
import matplotlib.pyplot as plt


class SoilColumn:
    """
    1D soil heat diffusion using the heat equation.
    ∂T/∂t = α ∂²T/∂z²
    """

    def __init__(self, depth=1.0, layers=50, dz=None, dt=60, total_time=12*3600,
                 initial_temp=15.0, surface_loss=0.005, diffusivity=1e-6):
        """
        Parameters:
            depth (m): total soil depth
            layers (int): number of vertical layers
            dz (m): optional layer thickness (overrides depth/layers)
            dt (s): timestep
            total_time (s): total simulation time
            initial_temp (°C): uniform initial soil temperature
            surface_loss (°C/s): cooling rate at the surface (simulates radiation loss)
            diffusivity (m²/s): soil thermal diffusivity
        """
        self.layers = layers
        self.dz = depth / layers if dz is None else dz
        self.dt = dt
        self.steps = int(total_time / dt)
        self.diffusivity = diffusivity
        self.surface_loss = surface_loss

        # Initialize temperature profile
        self.state = np.full(layers, initial_temp)

    def step(self):
        """Advance one timestep with explicit FTCS scheme."""
        new_state = self.state.copy()

        # Heat diffusion in interior nodes
        for i in range(1, self.layers - 1):
            new_state[i] = (self.state[i] +
                            self.diffusivity * self.dt / self.dz**2 *
                            (self.state[i+1] - 2*self.state[i] + self.state[i-1]))

        # Boundary conditions
        # Surface loses heat to the air
        new_state[0] = self.state[0] - self.surface_loss * self.dt
        # Deep soil (bottom layer) stays constant (infinite reservoir assumption)
        new_state[-1] = self.state[-1]

        self.state = new_state

    def run(self):
        """Run full simulation, return snapshots."""
        snapshots = []
        for step in range(self.steps):
            if step % 60 == 0:  # save every simulated hour
                snapshots.append(self.state.copy())
            self.step()
        return np.array(snapshots)


def simulate_and_plot():
    # Soil thermal diffusivity values (m²/s)
    soils = {
        "Sand": 1.2e-6,
        "Clay": 0.8e-6,
        "Loam": 1.0e-6
    }

    plt.figure(figsize=(8, 5))

    for soil, alpha in soils.items():
        model = SoilColumn(
            depth=1.0, layers=50,
            dt=60, total_time=12*3600,  # 12 hours
            initial_temp=15.0,
            surface_loss=0.0002,  # °C/s ~ 0.7°C/hour
            diffusivity=alpha
        )
        results = model.run()

        surface_temps = results[:, 0]  # top layer
        hours = np.arange(len(surface_temps))
        plt.plot(hours, surface_temps, label=soil)

    plt.xlabel("Hours after sunset")
    plt.ylabel("Surface Temperature (°C)")
    plt.title("Nighttime Soil Cooling in Different Soil Types")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    simulate_and_plot()
