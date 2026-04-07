SOLAR SYSTEM SIMULATION
University of Edinburgh — Computer Simulation (PHYS08026)
Author: Siddhant Sharma (S2730191)

FILES
-----
main.py           Entry point. Change RUN_MODE to select experiment.
simulation.py     Core Simulation class (physics, integrators, energy, periods).
bodies.py         Body class (mass, position, velocity, acceleration).
experiment_1.py   Experiment 1: Orbital periods vs NASA reference values.
experiment_2.py   Experiment 2: Energy conservation across three integrators.
experiment_3.py   Experiment 3: Satellite to Mars from L2 Lagrange point.
data/planets.json Sun + 8 planets: masses, orbital radii (SI), colours.
output/           Auto-created. Stores CSVs and figures from each experiment.

REQUIREMENTS
------------
Python 3.10+
pip install numpy matplotlib pandas

HOW TO RUN
----------
In your CLI, change directory to parent folder where the files have been unzipped
Then, run the following

python main.py --mode <mode>

where <mode> is one of:

    sim    default solar system animation
    exp1   Experiment 1: Orbital Periods
    exp2   Experiment 2: Energy Conservation
    exp3   Experiment 3: Satellite to Mars

EXAMPLES
--------
    python main.py --mode sim
    python main.py --mode exp1
    python main.py --mode exp2
    python main.py --mode exp3

For help:
    python main.py --help

NOTE
----
If --mode is not provided, the script falls back to the
RUN_MODE variable set at the bottom of main.py (default: "sim").

RUN_MODE options:
    "sim"   — Solar system animation (Beeman integrator, dt = T/1000)
    "exp1"  — Experiment 1: Orbital Periods
    "exp2"  — Experiment 2: Energy Conservation
    "exp3"  — Experiment 3: Satellite to Mars

EXPERIMENTS
-----------
Experiment 1:
  Beeman integrator run for 170 years at 5 time-step fractions (T/200 to T/1000).
  Simulated periods compared to NASA values.
  Output: output/exp_1_results/orbital_periods_dt_comparison.csv

Experiment 2:
  All three integrators run for 50 years. ΔE/E₀ logged every step.
  Beeman and Euler-Cromer are symplectic (bounded energy).
  Direct Euler drifts to +0.61% over 50 years.
  Output: output/exp_2_results/exp2_combined.png
          output/exp_2_results/exp2_individual.png
          output/exp_2_results/energy_summary.txt
  THIS TAKES A LONG TIME, REDUCE RUN OR TIME-STEP FRACTION FOR FASTER RESULTS

Experiment 3:
  Satellite launched from Sun-Earth L2 (~1.5 million km beyond Earth).
  Parameter sweep: Δv in [1.0, 1.6] × Hohmann_dv, θ in [-60°, 60°].
  Mars detection threshold: 0.02 AU.
  Earth return threshold:   0.01 AU.
  Scoring: S = 0.65*(d/d_max) + 0.35*(t/t_max). No fuel modelled.
  Best result: Δv=3685 m/s, θ=26.4°, closest approach 0.0003 AU, 423 days.
  Output: output/mars_experiment/parameter_sweep.csv

  THIS TAKES A LONG TIME. REDUCE PARAMETER SWEEP FOR FASTER RESULTS

NOTES
-----
- All code is commented, some for explanation and some for self-reference.
- An output folder is generated automatically in the same directory as the 
  rest of the files. This is done when the experiments are run.
- The simulation is not optimised for speed.
- Docstrings are given for all functions and classes and files.
- This codebase is maintained by me and is a public repo on GitHub, as I intend to
  continue to improve it later.

https://github.com/shadowsid12/Solar-System-Model