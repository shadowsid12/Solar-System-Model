# Solar System Simulation
A two-dimensional N-body gravitational simulation of the solar system, built for the University of Edinburgh Computer Simulation project (PHYS08026).

## File Structure

| File | Description |
|---|---|
| `main.py` | Entry point. Set `RUN_MODE` to select what to run. |
| `simulation.py` | Core `Simulation` class — physics, integrators, energy, period detection. |
| `bodies.py` | `Body` dataclass — physical and kinematic state of one celestial body. |
| `experiment_1.py` | Experiment 1: Orbital Periods vs NASA reference values. |
| `experiment_2.py` | Experiment 2: Energy conservation across three integrators. |
| `experiment_3.py` | Experiment 3: Satellite to Mars from L2. |
| `data/planets.json` | Sun + 8 planets: masses, orbital radii (SI), display colours. |
| `output/` | Auto-created. Stores CSVs and plots from each experiment. |

## Requirements

```bash
pip install numpy matplotlib pandas
```

## Running

Set `RUN_MODE` in `main.py`, then run:

```bash
python main.py
```

| `RUN_MODE` | Action |
|---|---|
| `"sim"` | Default solar system animation |
| `"exp1"` | Experiment 1: Orbital Periods |
| `"exp2"` | Experiment 2: Energy Conservation |
| `"exp3"` | Experiment 3: Satellite to Mars |

## Physics

### N-body Gravity

Newton's law gives the acceleration on body $j$:

```
a_j = G * sum_{i≠j}  m_i / |r_ij|^2 * r_hat_ij
```

Forces are equal and opposite (Newton's 3rd law), but accelerations differ because masses differ. Computed via vectorised NumPy einsum over the (N×N×2) displacement tensor — ~9x faster than a Python pair loop.

### Integrators

**Beeman** (default — used for all experiments):
```
r(t+dt) = r(t) + v(t)*dt + (1/6)[4a(t) - a(t-dt)] * dt²
a(t+dt) = recompute at new positions
v(t+dt) = v(t) + (1/6)[2a(t+dt) + 5a(t) - a(t-dt)] * dt
```

**Euler-Cromer** (symplectic — energy bounded, no secular drift):
```
v(t+dt) = v(t) + a(t)*dt
r(t+dt) = r(t) + v(t+dt)*dt     ← new velocity
```

**Direct Euler** (non-symplectic — energy drifts, included for comparison):
```
r(t+dt) = r(t) + v(t)*dt        ← old velocity
v(t+dt) = v(t) + a(t)*dt
```

### Initial Conditions

All planets start on the positive x-axis at their orbital radius with Keplerian circular speed in +y. The centre-of-mass velocity is subtracted from all bodies to give zero net momentum.

### Default Time Step

`dt = T_earth / 1000 ≈ 8.77 hours` — balances accuracy and speed.

---

## Experiment 1: Orbital Periods

Beeman is run for 170 years at five time-step fractions (T/200 to T/1000). Simulated periods are compared to NASA sidereal values. Mercury benefits most from finer dt; outer planet errors are dominated by N-body perturbations from Jupiter and are insensitive to dt.

## Experiment 2: Energy Conservation

All three integrators run for 50 years. Fractional energy change `ΔE/E₀` is logged every step. Beeman and Euler-Cromer stay bounded (symplectic); Direct Euler drifts to +0.61% over 50 years. Two centred rolling means (1-year and 3-year windows) reveal oscillatory structure.

Output saved to `output/exp_2_results/`: `exp2_combined.png`, `exp2_individual.png`, `energy_summary.txt`.

## Experiment 3: Satellite to Mars

Satellite is placed at the Sun-Earth L2 Lagrange point (~1.5 million km beyond Earth) with the heliocentric parking velocity required to co-rotate with Earth (~30,283 m/s). A delta-v burn is applied in direction θ from Earth's prograde.

**Parameter sweep:** `Δv ∈ [1.0, 1.6] × Hohmann_dv`, `θ ∈ [−60°, 60°]`

**Mars detection:** satellite must come within 0.02 AU of Mars to count as reaching it.

**Return detection:** satellite must come within 0.01 AU of Earth after the fly-past.

**Scoring:** `S = 0.65*(d/d_max) + 0.35*(t/t_max)` — lower is better. No fuel is modelled.

**Best result:** Δv = 3685 m/s, θ = 26.4°, closest approach 0.0003 AU (~45,000 km), journey 423 days.

Output saved to `output/mars_experiment/parameter_sweep.csv`.