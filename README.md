# Solar System Simulation

A two-dimensional N-body gravitational simulation of the solar system, built as part of a Computer Simulation project.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point. Set `RUN_MODE` to select the default simulation or an experiment. |
| `simulation.py` | Core `Simulation` class — physics, integrators, energy, period detection. |
| `bodies.py` | `Body` class representing a single celestial body. |
| `experiments.py` | Self-contained functions for Experiments 1, 2, and 3. |
| `data/planets.json` | Input data — masses, orbital radii, colours for the Sun and all eight planets in SI units. |
| `output/` | Auto-created at runtime. Stores `energy_beeman.csv`. |

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

```bash
pip install numpy matplotlib
```

## How to Run

All commands from the `solar_system/` directory. Set `RUN_MODE` at the top of `main.py`, then run:

```bash
python main.py
```

### Default Simulation

```python
RUN_MODE = "sim"
```

Animates all eight planets orbiting the Sun using the Beeman integrator. Prints orbital periods to the terminal and writes energy data to `output/energy_beeman.csv`.

### Experiment 1 — Orbital Periods

```python
RUN_MODE = "exp1"
```

Compares simulated orbital periods against NASA reference values at three time step sizes. Produces a bar chart of percentage error per planet.

### Experiment 2 — Energy Conservation

```python
RUN_MODE = "exp2"
```

Runs Beeman, Euler-Cromer, and Direct Euler for 5 Earth years. Produces a combined energy plot and per-integrator subplots with rolling mean overlays.

### Experiment 3 — Satellite to Mars

```python
RUN_MODE = "exp3"
```

Searches over a range of launch speeds to find trajectories achieving a close Mars fly-past. Reports minimum distance, journey time, and whether the satellite returns to Earth.

## Design

- The simulation is a true many-body problem — gravitational interactions between all pairs of bodies are computed at every step. Adding new planets requires only a new entry in `data/planets.json`.
- Three integrators are implemented: **Beeman** (used for all experiments), **Euler-Cromer**, and **Direct Euler**.
- The default time step of `EARTH_YEAR / 1000` (~8.77 hours) gives period errors under 0.15% for all planets vs NASA values.