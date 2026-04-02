"""
main.py
-------
Entry point. Runs the default solar system simulation with a matplotlib
FuncAnimation. Experiments are handled in experiments.py.
"""

from pathlib import Path
from collections import deque

import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg is unavailable
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulation import Simulation

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

EARTH_YEAR = 365.25 * 24 * 3600   # seconds
SUN_MASS   = 1.989e30              # kg
DATA_FILE  = Path(__file__).parent / "data" / "planets.json"

# ------------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------------

DT             = EARTH_YEAR / 1000   # time step (~8.77 hours)
STEPS_PER_FRAME = 10                  # sim steps advanced per animation frame
TRAIL_LENGTH   = 300                 # number of past positions kept per body
TOTAL_YEARS    = 170                 # run long enough to capture Neptune (~164.8 yr)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_default_simulation():
    # --- Setup the simulation ---
    sim = Simulation(dt=DT, integrator="beeman")
    sim.load_bodies_from_json(str(DATA_FILE))
    sim.initialise_bodies(sun_mass=SUN_MASS)

    total_steps = int(TOTAL_YEARS * EARTH_YEAR / DT)
    energy_log_interval = 50    # log energy every N steps

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black") # subplot allows for more control over axes and figure
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    # Axis limits in AU — just beyond Neptune's orbit
    AU = 1.496e11
    lim = 32.0 #Neptune's orbital radius is about 30AU
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white") # the little notches on the axis
    for spine in ax.spines.values():
        spine.set_edgecolor("white") # axis spine is the main axis line itself

    # Time label
    time_text = ax.text(
        0.02, 0.96, "", transform=ax.transAxes,
        color="white", fontsize=10, va="top"
    )
    """
    transform = tansAxes fixes the text label to position on the axes, not figure
    If I zoom out, the text label will stay in the same place because of this (placed in top right for now)
    """

    # --- Create one dot + one trail line per body ---
    dots  = []
    lines = []
    trails = []   # deque of (x_AU, y_AU) per body

    for body in sim.bodies:
        # Dot — slightly larger for Sun
        size = 12 if body.name == "Sun" else 5
        dot, = ax.plot([], [], "o", color=body.colour, markersize=size,
                       label=body.name, zorder=3) # z-order is basically vertical stacking - dots are above trail is above canvas
        dots.append(dot)

        # Trail line
        line, = ax.plot([], [], "-", color=body.colour, linewidth=0.8,
                        alpha=0.5, zorder=2)
        lines.append(line)

        # Trail history
        trails.append(deque(maxlen=TRAIL_LENGTH))

        """
        deque is a data structure that stores a fixed-length collection of items. once max_len is reached, any new entries will
        replace the oldest item in the deque
        """

    ax.legend(loc="upper right", fontsize=7, facecolor="#111111",
              labelcolor="white", framealpha=0.7)

    # Track whether all periods have been found
    planet_names = [b.name for b in sim.bodies[1:] if not b.is_satellite]
    periods_done: bool = False
    step_counter: list[int] = [0]   # mutable so update() can modify it

    # --- Animation update function ---
    def update(frame):
        nonlocal periods_done

        for _ in range(STEPS_PER_FRAME):
            if step_counter[0] >= total_steps:
                return dots + lines + [time_text]

            sim.step()
            step_counter[0] += 1

            if step_counter[0] % energy_log_interval == 0:
                sim.log_energy()

        # Check if all periods found (only print once)
        if not periods_done and all(n in sim.periods for n in planet_names):
            periods_done = True
            sim.print_periods()

        # Update dots and trails
        for i, b in enumerate(sim.bodies):
            x_au = b.position[0] / AU
            y_au = b.position[1] / AU

            dots[i].set_data([x_au], [y_au])

            trails[i].append((x_au, y_au))
            if len(trails[i]) > 1:
                tx, ty = zip(*trails[i])
                lines[i].set_data(tx, ty)

        # Update time display
        years = sim.time / EARTH_YEAR
        time_text.set_text(f"t = {years:.2f} yr")

        return dots + lines + [time_text]

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_steps // STEPS_PER_FRAME,
        interval=20,        # ms between frames (~50 fps target)
        blit=True,
        repeat=False,
    )

    plt.title("Solar System Simulation", color="white", fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- After animation closes, write energy to file ---
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    sim.write_energy_to_file(str(output_dir / "energy_beeman.csv"))
    print(f"\nEnergy log written to output/energy_beeman.csv  ({len(sim.energy_log)} entries)")


def run_experiment_1():
    from experiment_1 import run_experiment_1 as exp1
    exp1(str(DATA_FILE), dt_fractions=[200, 500, 1000])


def run_experiment_2():
    from experiment_2 import run_experiment_2 as exp2
    exp2(str(DATA_FILE), dt=DT, num_years=5)


def run_experiment_3():
    from experiment_3 import run_experiment_3 as exp3
    import numpy as np
    launch_speeds = list(np.linspace(2000, 4500, 25))  # m/s above Earth's orbital velocity
    exp3(str(DATA_FILE), launch_speeds, dt=DT)


# Change RUN_MODE to select what to run:
#   "sim"  — default solar system animation
#   "exp1" — Experiment 1: Orbital Periods
#   "exp2" — Experiment 2: Energy Conservation
#   "exp3" — Experiment 3: Satellite to Mars
RUN_MODE = "sim"

if __name__ == "__main__":
    if RUN_MODE == "exp1":
        run_experiment_1()
    elif RUN_MODE == "exp2":
        run_experiment_2()
    elif RUN_MODE == "exp3":
        run_experiment_3()
    else:
        run_default_simulation()