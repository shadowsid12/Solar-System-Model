# Solar System Simulation

A two-dimensional N-body gravitational simulation of the solar system, built as part of the University of Edinburgh Computer Simulation project.

## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point. Set `RUN_MODE` to select what to run. |
| `simulation.py` | Core `Simulation` class — physics, integrators, energy, period detection. |
| `bodies.py` | `Body` class — stores physical and kinematic state of one celestial body. |
| `experiment_1.py` | Experiment 1: Orbital Periods. |
| `experiment_2.py` | Experiment 2: Energy Conservation. |
| `experiment_3.py` | Experiment 3: Satellite to Mars. |
| `data/planets.json` | Input data — masses, orbital radii (SI), colours for Sun + 8 planets. |
| `output/` | Auto-created. Stores `energy_beeman.csv` and `mars_experiment/parameter_sweep.csv`. |

## Requirements

```bash
pip install numpy matplotlib pandas
```

## How to Run

Set `RUN_MODE` in `main.py`, then:

```bash
python main.py
```

| `RUN_MODE` | Action |
|---|---|
| `"sim"` | Default solar system animation (Section 3) |
| `"exp1"` | Experiment 1: Orbital Periods |
| `"exp2"` | Experiment 2: Energy Conservation |
| `"exp3"` | Experiment 3: Satellite to Mars |

---

## Physics and Mathematics

### Numerical Integration

Newton's second law **F** = m**a** is decomposed into two first-order ODEs:

```
dr/dt = v
dv/dt = a = F/m
```

Three schemes are implemented:

**Beeman** (primary — used for all experiments):
```
r(t+dt) = r(t) + v(t)*dt + (1/6)[4a(t) - a(t-dt)] * dt²
a(t+dt) = compute_accelerations() at new positions
v(t+dt) = v(t) + (1/6)[2a(t+dt) + 5a(t) - a(t-dt)] * dt
```

**Euler-Cromer** (symplectic):
```
v(t+dt) = v(t) + a(t)*dt
r(t+dt) = r(t) + v(t+dt)*dt        ← uses NEW velocity
```

**Direct Euler** (non-symplectic, for comparison in Experiment 2):
```
r(t+dt) = r(t) + v(t)*dt           ← uses OLD velocity
v(t+dt) = v(t) + a(t)*dt
```

### Gravitational Acceleration

For each unique pair (i, j), accelerations are computed using Newton's law of gravitation and his third law:

```
a_j = G * mᵢ / |rᵢⱼ|² * r̂ᵢⱼ
a_i = G * mⱼ / |rᵢⱼ|² * (-r̂ᵢⱼ)
```

where **r**ᵢⱼ = **r**ᵢ − **r**ⱼ, r̂ᵢⱼ is the unit vector, and G = 6.6743 × 10⁻¹¹ m³ kg⁻¹ s⁻². Forces are equal and opposite, but accelerations differ because masses differ.

### Initial Conditions

Each planet starts at orbital radius r on the positive x-axis with Keplerian circular speed in +y:

```
r(0) = (orbital_radius, 0)
v(0) = (0, sqrt(G * M_sun / r))
```

The centre-of-mass velocity `v_com = Σ(mᵢvᵢ) / Σmᵢ` is subtracted from every body so total system momentum is exactly zero, preventing barycentre drift.

### System Energy

```
KE = Σᵢ ½ mᵢ |vᵢ|²
PE = −G Σᵢ₍ᵢ﹤ⱼ₎ mᵢmⱼ / |rᵢⱼ|
E  = KE + PE
```

Summing PE over unique pairs (i < j) avoids double-counting.

---

## Experiment 1: Orbital Periods

Orbital periods are detected by tracking each planet's cumulative angle relative to the Sun:

```
θ(t) = atan2(y − y_sun, x − x_sun)
```

The angular step is normalised to (−π, π] each time step to handle the atan2 discontinuity at ±π:

```
Δθ = (θ(t) − θ(t−dt) + π) mod 2π − π
```

A full orbit is complete when the cumulative angle first reaches 2π.

The experiment runs at three time step sizes (`dt = year/200`, `year/500`, `year/1000`) to show how accuracy improves. Mercury has the largest error at coarse dt because it completes the most orbits and has the most curved path — discretisation error compounds fastest.

---

## Experiment 2: Energy Conservation

The fractional energy change is tracked over 5 years:

```
ΔE/E₀ = (E(t) − E(0)) / |E(0)|
```

Two rolling mean overlays are applied to each integrator's energy trace to reveal oscillatory structure:

- **First rolling mean** (window = 1 year of steps): smooths fast oscillations driven by Mercury and Earth
- **Second rolling mean** (window = 3 years of steps): applied to the first smooth signal, reveals the slower envelope

Both use a **centred** window to avoid phase lag. Rolling mean at index i:
```
mean[i] = average of data[i − half : i + half + 1]
```

**Why Beeman and Euler-Cromer conserve energy**: both are symplectic (or symplectic-like), preserving a modified Hamiltonian exactly — energy oscillates around a fixed mean with no drift. **Why Direct Euler does not**: it is non-symplectic — position uses the old velocity and velocity uses the old acceleration, making the updates inconsistent in phase space and injecting energy each step.

---

## Experiment 3: Satellite to Mars

### Launch Conditions — Sun-Earth L1 Lagrange Point

The satellite is launched from the Sun-Earth L1 Lagrange point — the unstable equilibrium where gravitational and centrifugal forces balance in the co-rotating frame. This is where JWST sits.

**L1 distance from Earth** (Hill sphere approximation):
```
r_L1 = R × (M_earth / 3·M_sun)^(1/3) ≈ 1.496 million km
```

**L1 heliocentric parking speed** (must co-rotate with Earth):
```
v_L1 = ω_earth × (R − r_L1) ≈ 29,488 m/s
```
This is ~301 m/s slower than Earth's orbital speed — Earth's gravity compensates, allowing co-rotation at a closer solar radius.

**Simulation placement:**
```
pos_L1 = earth.position − earth_r̂ × r_L1     (Sun side of Earth)
vel_L1 = v_L1 × earth_v̂                      (prograde)
```

**Burn direction** rotated θ from Earth's prograde:
```
burn_dir = cos θ · v̂_earth + sin θ · r̂_earth
v_sat    = vel_L1 + Δv × burn_dir
```

| θ | Direction |
|---|---|
| 0° | Purely prograde (minimum energy toward Mars) |
| 90° | Purely radial outward |
| 180° | Retrograde |

**Hohmann Δv from L1 to Mars** (theoretical minimum energy single burn):
```
v_transfer = sqrt(G·M_sun · (2/r_L1 − 1/((r_L1 + r_mars)/2)))
Δv_Hohmann = v_transfer − v_L1 ≈ 3475 m/s
```
The parameter sweep is centred on this value.

### Rocket Equation

The Tsiolkovsky rocket equation for a single impulsive burn:

```
m_wet  = m_dry × exp(Δv / v_exhaust)
m_fuel = m_wet − m_dry = m_dry × (exp(Δv / v_exhaust) − 1)
```

| Parameter | Value |
|---|---|
| `m_dry` (payload) | 2000 kg |
| `v_exhaust` | 4400 m/s (Isp ≈ 450 s, chemical propulsion) |
| `Δv` | `extra_speed` (m/s) |
| `m_wet` | total launch mass = m_dry + m_fuel |

The Hohmann transfer Δv from Earth to Mars orbit is ~2960 m/s — the theoretical minimum-energy single-burn trajectory for circular coplanar orbits.

### Return-to-Earth Detection

A satellite is considered returned only if **both** conditions hold simultaneously, and only **after** the Mars fly-past:

```
1. |r_sat − r_earth| < 0.01 AU          (radial proximity)
2. |angle_sat − angle_earth| < 5°       (angular proximity)

angle = atan2(y, x)
delta_angle = |(angle_sat − angle_earth + π) mod 2π − π|
```

Condition 2 is critical: without it, any trajectory passing through ~1 AU after ~1 year falsely satisfies condition 1 because Earth has orbited back to a similar radius — but the satellite may be on the opposite side of the Sun.

### Trajectory Scoring

All trajectories are ranked by a weighted normalised score (lower = better):

```
score = 0.5 × (d / d_max) + 0.3 × (t / t_max) + 0.2 × (f / f_max)
```

| Term | Metric | Weight | Rationale |
|---|---|---|---|
| d / d_max | Closest Mars approach (AU) | 0.5 | Primary mission goal |
| t / t_max | Journey time (days) | 0.3 | Shorter = less operational risk |
| f / f_max | Fuel mass (kg) | 0.2 | Lower = cheaper launch |

Each metric is normalised to [0, 1] across all candidates so differences in units do not bias the result.

**Reference**: NASA's Perseverance rover (2020–2021) travelled Earth to Mars in **203 days** using a Hohmann-like transfer timed to a real planetary alignment window.