"""
experiment_2.py
---------------
Experiment 2: Energy Conservation & Alternative Integration Methods

Runs Beeman, Euler-Cromer, and Direct Euler for a fixed duration.
Produces a combined energy plot and per-integrator subplots with rolling
mean overlays to reveal oscillatory structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation

EARTH_YEAR = 365.25 * 24 * 3600
SUN_MASS   = 1.989e30

def run_experiment_2(data_file: str, dt: float, num_years: int = 5) -> None:
    """
    Experiment 2: Energy Conservation & Alternative Integration Methods
    -------------------------------------------------------------------
    Runs three separate simulations (Beeman, Euler-Cromer, Direct Euler)
    with the same dt and initial conditions. Logs total energy every step.

    Produces:
      - A combined plot of fractional energy change vs time for all three methods
      - A dedicated Beeman-only plot with a narrow y-axis to show fine detail

    Parameters
    ----------
    data_file : path to planets.json
    dt        : time step in seconds (same for all three runs)
    num_years : simulation duration in Earth years
    """
    integrators = ("beeman", "euler_cromer", "direct_euler")
    labels      = ("Beeman", "Euler-Cromer", "Direct Euler")
    colours     = ("steelblue", "darkorange", "firebrick")
    total_steps = int(num_years * EARTH_YEAR / dt)

    # Store (times_yr, frac_energy_change) per integrator
    energy_data = {}

    for integrator in integrators:
        sim = Simulation(dt=dt, integrator=integrator)
        sim.load_bodies_from_json(data_file)
        sim.initialise_bodies(sun_mass=SUN_MASS)

        e0 = sim.total_energy()   # baseline energy at t=0

        times = []
        frac_changes = []

        for step in range(total_steps):
            sim.step()
            e = sim.total_energy()
            times.append(sim.time / EARTH_YEAR)
            frac_changes.append((e - e0) / abs(e0))

        energy_data[integrator] = (times, frac_changes)
        print(f"{integrator:<15}  final dE/E0 = {frac_changes[-1]:+.6e}")

    # --- Plot 1: all three integrators on one axes ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    for integrator, label, colour in zip(integrators, labels, colours):
        times, frac = energy_data[integrator]
        ax1.plot(times, frac, label=label, color=colour, linewidth=1.2)

    ax1.set_xlabel("Time (Earth years)")
    ax1.set_ylabel("Fractional energy change  (E - E₀) / |E₀|")
    ax1.set_title("Experiment 2: Energy Conservation — All Three Integrators")
    ax1.legend()
    ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: one subplot per integrator, y-axis set to actual min/max ---
    fig2, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    def rolling_mean(data: list, window: int) -> np.ndarray:
        """
        Compute centred rolling mean with the given window size.
        Edges are handled by shrinking the window so every point gets a value.
        """
        arr = np.array(data)
        out = np.empty_like(arr)
        half = window // 2
        for i in range(len(arr)):
            lo = max(0, i - half)
            hi = min(len(arr), i + half + 1)
            out[i] = arr[lo:hi].mean()
        return out

    # Window size in steps: ~1 Earth year worth of steps smooths out
    # short-period oscillations (Mercury ~88 days, Earth 365 days)
    fast_window = int(EARTH_YEAR / dt)   # 1-year rolling mean
    slow_window = fast_window * 3        # 3-year rolling mean of that

    for ax, integrator, label, colour in zip(axes, integrators, labels, colours):
        times, frac = energy_data[integrator]
        arr = np.array(frac)

        # First smoothing: averages out the fast oscillation
        smooth_1 = rolling_mean(frac, fast_window)

        # Second smoothing: applied to the already-smoothed signal,
        # reveals the slower underlying trend
        smooth_2 = rolling_mean(smooth_1.tolist(), slow_window)

        # Raw signal
        ax.plot(times, arr, color=colour, linewidth=0.8, alpha=0.5, label="Raw")

        # First rolling mean
        ax.plot(times, smooth_1, color="white", linewidth=1.2,
                linestyle="--", alpha=0.8, label=f"Rolling mean ({fast_window} steps)")

        # Second rolling mean (trend)
        ax.plot(times, smooth_2, color="yellow", linewidth=1.6,
                linestyle="-", label=f"Rolling mean of mean ({slow_window} steps)")

        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")

        # y-axis bounds from raw data
        y_min, y_max = arr.min(), arr.max()
        pad = (y_max - y_min) * 0.15 or abs(y_max) * 0.15
        ax.set_ylim(y_min - pad, y_max + pad)

        ax.set_ylabel("(E - E₀) / |E₀|", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time (Earth years)")
    fig2.suptitle("Experiment 2: Energy Conservation — Individual Integrators", fontsize=12)
    plt.tight_layout()
    plt.show()