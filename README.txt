================================================================================
Solar System Simulation — README
Author: Siddhant Sharma
================================================================================

--------------------------------------------------------------------------------
FILES INCLUDED
--------------------------------------------------------------------------------

main.py
    Entry point. Set RUN_MODE at the top to select the default simulation or
    an experiment, then run: python main.py

simulation.py
    Core Simulation class. Handles loading bodies, initialising positions and
    velocities, stepping forward in time, computing gravitational accelerations,
    detecting orbital periods, and logging total system energy.

bodies.py
    Body class. Stores physical properties (mass, orbital radius) and kinematic
    state (position, velocity, current and previous acceleration) for one body.

experiment_1.py
    Experiment 1: Orbital Periods. Runs the Beeman simulation at multiple time
    step sizes and compares simulated periods against NASA reference values.

experiment_2.py
    Experiment 2: Energy Conservation. Runs all three integrators and plots
    fractional energy change vs time, with rolling mean overlays.

experiment_3.py
    Experiment 3: Satellite to Mars. Sweeps over launch speeds and angles,
    selects the best trajectory by weighted score, and produces plots and an
    animation.

data/planets.json
    Input data: masses and orbital radii (SI units) and display colours for
    the Sun and all eight planets (Mercury to Neptune).

output/
    Created automatically at runtime. Stores energy_beeman.csv and the
    mars_experiment/parameter_sweep.csv.

--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Python 3.10 or later
numpy, matplotlib, pandas

    pip install numpy matplotlib pandas

--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

All commands from the solar_system/ directory.
Set RUN_MODE in main.py, then: python main.py

    RUN_MODE = "sim"   Default simulation (Section 3)
    RUN_MODE = "exp1"  Experiment 1
    RUN_MODE = "exp2"  Experiment 2
    RUN_MODE = "exp3"  Experiment 3

--------------------------------------------------------------------------------
PHYSICS AND MATHEMATICS
--------------------------------------------------------------------------------

NUMERICAL INTEGRATION
---------------------
Newton's second law F = ma is split into two first-order ODEs:

    dr/dt = v
    dv/dt = a = F/m

Three integration schemes are implemented:

Beeman (primary):
    r(t+dt) = r(t) + v(t)*dt + (1/6)[4a(t) - a(t-dt)] * dt^2
    a(t+dt) = computed from new positions
    v(t+dt) = v(t) + (1/6)[2a(t+dt) + 5a(t) - a(t-dt)] * dt

Euler-Cromer (symplectic):
    v(t+dt) = v(t) + a(t)*dt
    r(t+dt) = r(t) + v(t+dt)*dt        (uses NEW velocity)

Direct Euler (non-symplectic, for comparison):
    r(t+dt) = r(t) + v(t)*dt           (uses OLD velocity)
    v(t+dt) = v(t) + a(t)*dt

GRAVITATIONAL ACCELERATION
--------------------------
For each unique pair of bodies (i, j), the mutual accelerations are:

    a_j = G * m_i / |r_ij|^2 * r_hat_ij
    a_i = G * m_j / |r_ij|^2 * (-r_hat_ij)

where r_ij = r_i - r_j (vector from j to i), r_hat_ij is its unit vector,
and G = 6.6743e-11 m^3 kg^-1 s^-2. Note: forces are equal and opposite
(Newton's third law), but accelerations differ because masses differ.

INITIAL CONDITIONS
------------------
Each planet is placed at its orbital radius on the positive x-axis:

    r(0) = (orbital_radius, 0)

with Keplerian circular orbit speed in the +y direction:

    v(0) = (0, sqrt(G * M_sun / r))

The centre-of-mass velocity is then subtracted from every body (including
the Sun) so the total system momentum is exactly zero, preventing spurious
drift of the solar system barycentre.

ENERGY
------
Total kinetic energy:
    KE = sum_i (1/2) * m_i * |v_i|^2

Gravitational potential energy (unique pairs only, no double-counting):
    PE = -G * sum_{i<j} m_i * m_j / |r_ij|

Total energy E = KE + PE is conserved for a closed system. Beeman and
Euler-Cromer are symplectic-like and keep |dE/E| ~ 1e-6 over 5 years.
Direct Euler is non-symplectic: energy drifts by ~1.5% over 5 years.

--------------------------------------------------------------------------------
EXPERIMENT 1: ORBITAL PERIODS
--------------------------------------------------------------------------------

The simulation detects orbital periods by tracking the cumulative angle of
each planet relative to the Sun using atan2:

    theta(t) = atan2(y - y_sun, x - x_sun)

The angular step per time step is normalised to (-pi, pi] to handle the
atan2 wrap-around at +/-pi:

    delta = (theta(t) - theta(t-dt) + pi) mod 2*pi - pi

The period is recorded when the cumulative angle first reaches 2*pi.

The experiment is run at three time step sizes (dt = year/200, year/500,
year/1000) to show how accuracy improves with smaller dt. Mercury shows the
largest error at coarse dt because it has the shortest period and most
curved orbit — the fewest steps per orbit means discretisation error is
largest.

--------------------------------------------------------------------------------
EXPERIMENT 2: ENERGY CONSERVATION
--------------------------------------------------------------------------------

Each integrator is run for 5 Earth years. The fractional energy change is
plotted vs time:

    dE/E0 = (E(t) - E(0)) / |E(0)|

Two plots are produced:
  1. All three integrators on one axes — shows Direct Euler diverging
  2. One subplot per integrator with y-axis scaled to actual min/max, with
     two rolling mean overlays to reveal oscillatory structure:

     First rolling mean (window = 1 year of steps):
         Smooths short-period oscillations driven by Mercury and Earth.

     Second rolling mean (window = 3 years of steps):
         Applied to the first smooth signal — reveals the slower envelope.

     Both use a centred window so no phase lag is introduced.

Beeman and Euler-Cromer conserve energy well because they are symplectic
(or symplectic-like): they preserve a modified Hamiltonian exactly, so
energy oscillates around a fixed mean. Direct Euler is not symplectic —
it injects energy each step, causing monotonic upward drift.

--------------------------------------------------------------------------------
EXPERIMENT 3: SATELLITE TO MARS
--------------------------------------------------------------------------------

LAUNCH CONDITIONS — SUN-EARTH L1 LAGRANGE POINT
-------------------------------------------------
The satellite is launched from the L1 Lagrange point, the unstable equilibrium
on the Sun-Earth line where gravitational and centrifugal forces balance in the
co-rotating frame. This is where the James Webb Space Telescope sits.

L1 distance from Earth (Hill sphere approximation):

    r_L1 = R * (M_earth / 3*M_sun)^(1/3) ~= 1.496 million km

L1 heliocentric orbital speed (must co-rotate with Earth):

    v_L1 = omega_earth * (R - r_L1) ~= 29,488 m/s

where omega_earth = 2*pi / T_earth. This is ~301 m/s LESS than Earth's orbital
speed — Earth's gravity compensates, letting the satellite orbit more slowly
despite being closer to the Sun.

In the simulation frame (Earth at (R, 0) initially):

    pos_L1 = earth.position - earth_rhat * r_L1     (toward the Sun)
    vel_L1 = v_L1 * earth_vhat                      (prograde, tangential)

The delta-v burn is applied at angle theta from Earth's prograde direction:

    burn_dir = cos(theta)*earth_vhat + sin(theta)*earth_rhat
    v_sat    = vel_L1 + delta_v * burn_dir

    theta = 0 deg   purely prograde (minimum energy toward Mars)
    theta = 90 deg  purely radial outward (away from Sun)
    theta = 180 deg retrograde

Hohmann transfer delta-v from L1 to Mars (theoretical minimum energy):

    v_transfer_perihelion = sqrt(G*M_sun * (2/r_L1 - 1/((r_L1 + r_mars)/2)))
    delta_v_Hohmann       = v_transfer_perihelion - v_L1 ~= 3475 m/s

The sweep is centred on this Hohmann value.

ROCKET EQUATION
---------------
The Tsiolkovsky rocket equation gives the propellant mass needed for a
single impulsive burn of magnitude delta_v:

    m_wet   = m_dry * exp(delta_v / v_exhaust)
    m_fuel  = m_wet - m_dry = m_dry * (exp(delta_v / v_exhaust) - 1)

where:
    m_dry      = 2000 kg    (payload mass after burn)
    v_exhaust  = 4400 m/s   (chemical propulsion, Isp ~ 450 s)
    delta_v    = extra_speed (m/s)
    m_wet      = total launch mass (dry + fuel)

RETURN-TO-EARTH DETECTION
--------------------------
A satellite is considered to have returned to Earth only if BOTH conditions
hold simultaneously, and only AFTER the Mars fly-past:

    1. Distance to Earth < 0.01 AU  (~1.5 million km, ~Earth's Hill sphere)
    2. Angular separation from Earth < 5 degrees

    angle_sat   = atan2(y_sat, x_sat)
    angle_earth = atan2(y_earth, x_earth)
    delta_angle = |(angle_sat - angle_earth + pi) mod 2*pi - pi|

Condition 2 is essential: without it, any trajectory passing through ~1 AU
after ~1 year falsely triggers condition 1 because Earth has orbited back
to a similar radius, even if the satellite is on the opposite side of the Sun.

TRAJECTORY SCORING
------------------
Trajectories are ranked by a weighted normalised score (lower = better):

    score = 0.5 * (d / d_max) + 0.3 * (t / t_max) + 0.2 * (f / f_max)

where d = closest Mars approach (AU), t = journey time (days),
f = fuel mass (kg), and the denominators normalise each metric to [0, 1]
across all candidates so units do not bias the result.

Weights:
    0.5 — closest approach  (primary mission goal)
    0.3 — journey time      (shorter = less operational risk)
    0.2 — fuel mass         (lower = cheaper launch)

The Hohmann transfer delta-v from Earth to Mars is ~2960 m/s, giving the
theoretical minimum-energy trajectory. Our best trajectory uses a higher
delta-v because the simulation starts all planets on the positive x-axis,
which is not an optimal Mars launch window.

Reference: Perseverance (NASA, 2020-2021) travelled from Earth to Mars in
203 days using a Hohmann-like transfer orbit timed to a real launch window.

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------

- All experiments use the Beeman integrator unless stated otherwise.
- Default dt = EARTH_YEAR/1000 (~8.77 hours) gives period errors < 0.15%.
- Adding planets requires only a new entry in data/planets.json.