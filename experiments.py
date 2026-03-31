"""
experiments.py
--------------
Each function sets up and runs a specific experiment using Simulation.
All results (plots, prints, file writes) are handled here, not in main.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation
from bodies import Body

EARTH_YEAR = 365.25 * 24 * 3600
SUN_MASS   = 1.989e30

# NASA reference orbital periods in Earth years
NASA_PERIODS = {
    "Mercury": 0.2409,
    "Venus":   0.6152,
    "Earth":   1.0000,
    "Mars":    1.8809,
    "Jupiter": 11.862,
    "Saturn":  29.457,
    "Uranus":  84.011,
    "Neptune": 164.79,
}

# Years needed to simulate to capture at least one orbit of every planet
SIMULATION_YEARS = 170   # just beyond Neptune's 164.8 yr period


# ===========================================================================
# Experiment 1: Orbital Periods
# ===========================================================================

def run_experiment_1(data_file: str, dt_fractions: list[int] = None) -> None:
    """
    Experiment 1: Orbital Periods
    ------------------------------
    Runs the Beeman simulation at one or more time step sizes and compares
    the simulated orbital periods against NASA reference values.

    Produces:
      - A printed table of simulated vs NASA periods for each dt
      - A bar chart of percentage error per planet for each dt

    Parameters
    ----------
    data_file    : path to planets.json
    dt_fractions : list of steps-per-year to test, e.g. [200, 500, 1000]
                   Default: [200, 500, 1000]
    """
    if dt_fractions is None:
        dt_fractions = [200, 500, 1000]

    planet_names = list(NASA_PERIODS.keys())

    # results[fraction] = {planet_name: simulated_period_yr}
    results = {}

    for fraction in dt_fractions:
        dt = EARTH_YEAR / fraction
        sim = Simulation(dt=dt, integrator="beeman")
        sim.load_bodies_from_json(data_file)
        sim.initialise_bodies(sun_mass=SUN_MASS)

        total_steps = int(SIMULATION_YEARS * EARTH_YEAR / dt)

        for _ in range(total_steps):
            sim.step()
            if all(n in sim.periods for n in planet_names):
                break

        results[fraction] = {
            name: sim.periods[name] / EARTH_YEAR
            for name in planet_names
            if name in sim.periods
        }

        # Print table for this dt
        print(f"\n--- dt = EARTH_YEAR / {fraction}  ({dt/3600:.2f} hours) ---")
        print(f"{'Body':<10} {'Simulated (yr)':>15} {'NASA (yr)':>12} {'Error (%)':>10}")
        print("-" * 50)
        for name in planet_names:
            if name not in results[fraction]:
                print(f"{name:<10} {'NOT DETECTED':>15}")
                continue
            sim_yr  = results[fraction][name]
            ref     = NASA_PERIODS[name]
            error   = abs(sim_yr - ref) / ref * 100
            print(f"{name:<10} {sim_yr:>15.4f} {ref:>12.4f} {error:>9.3f}%")

    # --- Bar chart: percentage error per planet per dt ---
    x = np.arange(len(planet_names))
    width = 0.8 / len(dt_fractions)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, fraction in enumerate(dt_fractions):
        errors = []
        for name in planet_names:
            if name in results[fraction]:
                err = abs(results[fraction][name] - NASA_PERIODS[name]) / NASA_PERIODS[name] * 100
            else:
                err = float("nan")
            errors.append(err)

        offset = (i - len(dt_fractions) / 2 + 0.5) * width
        bars = ax.bar(x + offset, errors, width, label=f"dt = yr/{fraction}")

        # Label each bar with its value
        for bar, err in zip(bars, errors):
            if not np.isnan(err):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{err:.3f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(planet_names)
    ax.set_ylabel("Period error (%)")
    ax.set_title("Experiment 1: Orbital Period Error vs NASA Reference (Beeman)")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Experiment 2: Energy Conservation
# ===========================================================================

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
    fast_window = int(EARTH_YEAR / dt)       # 1-year rolling mean
    slow_window = fast_window * 3            # 3-year rolling mean of that

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


# ===========================================================================
# Experiment 3: Satellite to Mars
# ===========================================================================

def run_experiment_3(data_file: str, launch_speeds: list[float]) -> None:
    """
    Experiment 3: Satellite to Mars
    --------------------------------
    Launch a satellite from just above Earth and search over launch_speeds
    to find trajectories that achieve a close fly-past of Mars.

    For each speed records:
      - Minimum distance to Mars achieved
      - Journey time to closest approach
      - Whether the satellite returns within 1.1 AU of Earth

    Parameters
    ----------
    data_file    : path to planets.json
    launch_speeds: extra speeds (m/s) above Earth's orbital velocity to try
    """
    pass  # TODO


def _add_satellite(sim: Simulation, extra_speed: float) -> Body:
    """
    Create and add a satellite launched from just outside Earth's position.

    The satellite starts 1000 km above Earth (to avoid division-by-zero),
    inherits Earth's velocity, and receives extra_speed added in the +y
    direction (tangential to Earth's orbit).

    Parameters
    ----------
    extra_speed : additional speed in m/s added in the +y direction
    """
    pass  # TODO