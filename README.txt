SOLAR SYSTEM SIMULATION
University of Edinburgh — Computer Simulation (PHYS08026)
Author: Siddhant Sharma (S2730191)

FILES
-----
main.py           Entry point. Change RUN_MODE to select experiment.
simulation.py     Core Simulation class (physics, integrators, energy, periods).
bodies.py         Body dataclass (mass, position, velocity, acceleration).
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
Edit RUN_MODE in main.py, then:
    python main.py

RUN_MODE options:
    "sim"   — Solar system animation (Beeman, dt = T/1000)
    "exp1"  — Experiment 1: Orbital Periods
    "exp2"  — Experiment 2: Energy Conservation
    "exp3"  — Experiment 3: Satellite to Mars

EXPERIMENTS
-----------
Experiment 1:
  Beeman integrator run for 170 years at 5 time-step fractions (T/200 to T/1000).
  Simulated periods compared to NASA sidereal values.
  Output: output/exp_1_results/orbital_periods_dt_comparison.csv

Experiment 2:
  All three integrators run for 50 years. ΔE/E₀ logged every step.
  Beeman and Euler-Cromer are symplectic (bounded energy).
  Direct Euler drifts to +0.61% over 50 years.
  Output: output/exp_2_results/exp2_combined.png
          output/exp_2_results/exp2_individual.png
          output/exp_2_results/energy_summary.txt

Experiment 3:
  Satellite launched from Sun-Earth L2 (~1.5 million km beyond Earth).
  Parameter sweep: Δv in [1.0, 1.6] × Hohmann_dv, θ in [-60°, 60°].
  Mars detection threshold: 0.02 AU.
  Earth return threshold:   0.01 AU.
  Scoring: S = 0.65*(d/d_max) + 0.35*(t/t_max). No fuel modelled.
  Best result: Δv=3685 m/s, θ=26.4°, closest approach 0.0003 AU, 423 days.
  Output: output/mars_experiment/parameter_sweep.csv

NOTES
-----
- The centre-of-mass velocity is subtracted at initialisation so the
  system has zero net momentum (prevents barycentre drift).
- Beeman requires a(t-dt) at t=0, bootstrapped with a(0).
- Accelerations use vectorised NumPy einsum — ~9x faster than a pair loop.
- All planets start on the positive x-axis (not a real epoch).
  This means the initial geometry does not correspond to an optimal
  Mars launch window.