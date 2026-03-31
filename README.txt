================================================================================
Solar System Simulation — README
Author: Siddhant Sharma
================================================================================

--------------------------------------------------------------------------------
FILES INCLUDED
--------------------------------------------------------------------------------

main.py
    Entry point. Runs the default solar system simulation and experiments.
    Change RUN_MODE at the top of the file to select what to run.

simulation.py
    Core Simulation class. Handles loading bodies, initialising positions and
    velocities, stepping the simulation forward (Beeman, Euler-Cromer, Direct
    Euler), computing gravitational accelerations, detecting orbital periods,
    and logging total system energy.

bodies.py
    Body class. Represents a single celestial body (planet, sun, or satellite)
    and stores its physical properties and kinematic state.

experiments.py
    Functions for Experiments 1, 2, and 3. Each function is self-contained and
    produces its own output (printed tables and matplotlib plots).

data/planets.json
    Input data file. Contains masses, orbital radii, and display colours for
    the Sun and all eight planets (Mercury to Neptune) in SI units.

output/
    Directory created automatically when the simulation runs. Stores the energy
    log as a CSV file (energy_beeman.csv).

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Python 3.10 or later
numpy
matplotlib

Install dependencies with:
    pip install numpy matplotlib

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

All commands should be run from the solar_system/ directory.

--- Default Simulation (Section 3) ---

Open main.py and set:
    RUN_MODE = "sim"

Then run:
    python main.py

This will:
  - Animate the orbits of all eight planets around the Sun
  - Print orbital periods to the terminal once each planet completes one orbit
  - Write total system energy over time to output/energy_beeman.csv

Note: the simulation runs for 170 Earth years to cover Neptune's full orbit.
The animation uses the Beeman integration method with dt = 1 Earth year / 1000.

--- Experiment 1: Orbital Periods ---

Open main.py and set:
    RUN_MODE = "exp1"

Then run:
    python main.py

This compares simulated orbital periods against NASA reference values at three
different time step sizes (dt = year/200, year/500, year/1000) and produces a
bar chart of the percentage error per planet.

--- Experiment 2: Energy Conservation ---

Open main.py and set:
    RUN_MODE = "exp2"

Then run:
    python main.py

This runs all three integrators (Beeman, Euler-Cromer, Direct Euler) for 5
Earth years and produces:
  - A combined plot of fractional energy change vs time for all three methods
  - A separate subplot per integrator with y-axis scaled to actual min/max,
    with rolling mean overlays to reveal oscillatory structure

--- Experiment 3: Satellite to Mars ---

Open main.py and set:
    RUN_MODE = "exp3"

Then run:
    python main.py

This searches over a range of launch speeds to find trajectories that achieve
a close fly-past of Mars, and reports minimum distance, journey time, and
whether the satellite returns to Earth.

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------

- The simulation treats the solar system as a full many-body problem. Adding
  new planets requires only adding an entry to data/planets.json.

- All experiments use the Beeman integrator unless stated otherwise.

- The default dt of EARTH_YEAR/1000 (~8.77 hours) gives period errors of less
  than 0.15% for all planets compared to NASA reference values.