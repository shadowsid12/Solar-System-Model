"""
experiment_1.py
---------------
Experiment 1: Orbital Periods

Runs the Beeman simulation at multiple time step sizes and compares simulated
orbital periods against NASA reference values. Produces a printed table and
a bar chart of percentage error per planet per dt.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import Simulation

import csv
from pathlib import Path

EARTH_YEAR = 365.25 * 24 * 3600
SUN_MASS   = 1.989e30

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

    results = {}

    for fraction in dt_fractions:
        dt = EARTH_YEAR / fraction
        sim = Simulation(dt=dt, integrator="beeman")
        sim.load_bodies_from_json(data_file)
        sim.initialise_bodies(sun_mass=SUN_MASS)

        total_steps = int(SIMULATION_YEARS * EARTH_YEAR / dt)

        for i in range(total_steps):
            sim.step()
            if all(n in sim.periods for n in planet_names):
                break

        results[fraction] = {
            name: sim.periods[name] / EARTH_YEAR
            for name in planet_names
            if name in sim.periods
        }

        # Pretty-print table for this dt
        print(f"\n--- dt = EARTH_YEAR / {fraction}  ({dt/3600:.2f} hours) ---")
        print(f"{'Body':<10} {'Simulated (yr)':>15} {'NASA (yr)':>12} {'Error (%)':>10}")
        print("-" * 50)
        for name in planet_names:
            if name not in results[fraction]:
                print(f"{name:<10} {'NOT DETECTED':>15}")
                continue
            sim_yr = results[fraction][name]
            ref    = NASA_PERIODS[name]
            error  = abs(sim_yr - ref) / ref * 100
            print(f"{name:<10} {sim_yr:>15.4f} {ref:>12.4f} {error:>9.3f}%")

    # Export to CSV
    output_dir = Path(__file__).parent / "output" / "exp_1_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "orbital_periods_dt_comparison.csv"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header: Planet, then each DT fraction
        header = ["Planet"] + [f"dt_yr_{frac}" for frac in dt_fractions]
        writer.writerow(header)

        for name in planet_names:
            row = [name]
            for frac in dt_fractions:
                # Get simulated period or None if not detected
                val = results[frac].get(name, "N/A")
                row.append(val)
            writer.writerow(row)


    # Bar chart for percentage error per planet per dt
    x = np.arange(len(planet_names))
    width = 0.8 / len(dt_fractions)

    fig, ax = plt.subplots(figsize=(12, 5))

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
