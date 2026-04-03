"""
experiment_3.py
---------------
Experiment 3: Satellite to Mars

The satellite is parked at the Sun-Earth L2 Lagrange point — the unstable
equilibrium between Sun and Earth at ~1.496 million km from Earth toward the
Sun. L2 is where JWST sits and is a natural staging point for interplanetary
missions. The satellite is assumed to already be at L2 with the velocity
required to remain there (co-rotating with Earth). A delta-v burn then
sends it toward Mars.

Parameter sweep
---------------
- delta_v   : speed added at L2 (m/s); searched around the Hohmann
              transfer value L2 → Mars (~3475 m/s).
- theta_deg : direction of the burn in degrees from Earth's prograde
              direction (tangential to Earth's orbit):
                theta = 0   → purely prograde
                theta = 90  → purely radial outward (away from Sun)
                theta = 180 → retrograde

Outputs
-------
- CSV of all sweep results  (output/mars_experiment/parameter_sweep.csv)
- Static trajectory plot    (best scoring trajectory)
- Scatter plot              (journey time vs fuel, returning trajectories)
- Animation                 (best trajectory)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from simulation import Simulation
from bodies import Body

# ------------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------------

AU         = 1.496e11
DAY        = 24 * 3600
EARTH_YEAR = 365.25 * 24 * 3600
SUN_MASS   = 1.989e30
EARTH_MASS = 5.972e24
G          = 6.6743e-11

# ------------------------------------------------------------------
# L2 Lagrange point geometry
# ------------------------------------------------------------------
# Hill sphere approximation:
#     r_L2 = R * (M_earth / 3*M_sun)^(1/3)  ← distance from Earth
#
# Parking speed (heliocentric, co-rotating with Earth):
#     v_L2 = omega_earth * (R + r_L2)
#
# At L2 the satellite is further from the Sun, so Keplerian speed
# would be lower — but Earth's gravity pulls it forward, allowing
# it to orbit faster and keep up with Earth's angular velocity.

R_EARTH_SUN  = AU
R_L2_EARTH   = R_EARTH_SUN * (EARTH_MASS / (3 * SUN_MASS))**(1/3)  # ~1.496e9 m
R_L2_SUN     = R_EARTH_SUN + R_L2_EARTH
OMEGA_EARTH  = 2 * np.pi / EARTH_YEAR
V_L2         = OMEGA_EARTH * R_L2_SUN   # ~30283 m/s

# Hohmann delta-v from L2 to Mars
_V_TRANSFER  = np.sqrt(G * SUN_MASS * (2/R_L2_SUN - 1/((R_L2_SUN + 2.279e11)/2)))
HOHMANN_DV   = _V_TRANSFER - V_L2       # ~2681 m/s

# ------------------------------------------------------------------
# Mission parameters
# ------------------------------------------------------------------

PERSEVERANCE_JOURNEY_DAYS = 203.0
PAYLOAD_MASS              = 2000.0
V_EXHAUST                 = 4400.0

# ------------------------------------------------------------------
# Return-to-Earth thresholds
# ------------------------------------------------------------------

EARTH_RETURN_THRESHOLD_AU = 0.01
EARTH_RETURN_ANGLE_DEG    = 5.0

# ------------------------------------------------------------------
# Rocket equation
# ------------------------------------------------------------------

def fuel_mass(delta_v: float) -> float:
    """
    Tsiolkovsky rocket equation for single burn:
        m_fuel = m_dry * (exp(dv / v_exhaust) - 1)
        m_wet  = m_dry + m_fuel
    """
    return PAYLOAD_MASS * (np.exp(delta_v / V_EXHAUST) - 1)

# ------------------------------------------------------------------
# Satellite placement at L2
# ------------------------------------------------------------------

def add_satellite(sim: Simulation, delta_v: float, theta_deg: float) -> Body:
    """
    Place the satellite at L2 and apply a delta-v burn.

    L2 position
    -----------
    Along the Sun-Earth line, on the far side of Earth:
        pos_L2 = earth.position + earth_rhat * R_L2_EARTH
    where earth_rhat = earth.position / |earth.position| (Sun → Earth).

    L2 parking velocity
    -------------------
    Heliocentric, prograde (same direction as Earth's orbit):
        vel_L2 = V_L2 * earth_vhat
    where V_L2 = omega_earth * R_L2_SUN ≈ 30283 m/s.

    Burn
    ----
    delta_v is added in direction theta_deg from Earth's prograde:
        burn_dir = cos(θ)*earth_vhat + sin(θ)*earth_rhat
        v_sat = vel_L2 + delta_v * burn_dir

    Parameters
    ----------
    sim       : Simulation with planets initialised
    delta_v   : burn magnitude (m/s)
    theta_deg : burn direction in degrees from Earth's prograde
    """
    earth      = next(b for b in sim.bodies if b.name == "Earth")
    earth_rhat = earth.position / np.linalg.norm(earth.position)
    earth_vhat = earth.velocity / np.linalg.norm(earth.velocity)

    pos_L2   = earth.position + earth_rhat * R_L2_EARTH
    vel_L2   = V_L2 * earth_vhat

    theta_rad = np.radians(theta_deg)
    burn_dir  = np.cos(theta_rad) * earth_vhat + np.sin(theta_rad) * earth_rhat

    satellite = Body(
        name="Satellite",
        mass=PAYLOAD_MASS,
        orbital_radius=0.0,
        colour="white",
        is_satellite=True,
    )
    satellite.position = pos_L2
    satellite.velocity = vel_L2 + delta_v * burn_dir

    sim.add_body(satellite)
    accels = sim.compute_accelerations()
    satellite.acceleration      = accels["Satellite"]
    satellite.prev_acceleration = accels["Satellite"].copy()

    return satellite


# ------------------------------------------------------------------
# Weighted trajectory selector
# ------------------------------------------------------------------

def select_best(candidates: list[dict]) -> dict:
    """
    Weighted normalised score (lower = better):
        score = 0.5*(d/d_max) + 0.3*(t/t_max) + 0.2*(f/f_max)

    Weights:
        0.5 — closest Mars approach (primary mission goal)
        0.35 — journey time (shorter = less operational risk)
        0.15 — fuel mass (less = cheaper; correlated with time so down-weighted)
    """
    d_max = max(r["min_dist_au"]  for r in candidates)
    t_max = max(r["journey_days"] for r in candidates)
    f_max = max(r["fuel_kg"]      for r in candidates)

    def score(r):
        return (0.5 * r["min_dist_au"]  / d_max
              + 0.35 * r["journey_days"] / t_max
              + 0.15 * r["fuel_kg"]      / f_max)

    best = min(candidates, key=score)
    best["score"] = score(best)
    return best


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run_experiment_3(data_file: str, launch_speeds: list[float],
                     angles: list[float], dt: float) -> None:
    """
    Sweep delta-v and burn angle from L2. Records closest Mars approach,
    journey time, return status, and fuel for each combination.

    Parameters
    ----------
    data_file    : path to planets.json
    launch_speeds: delta-v values at L2 (m/s)
    angles       : burn angles in degrees from Earth's prograde direction
    dt           : time step (seconds)
    """
    MAX_YEARS   = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    output_dir = Path(__file__).parent / "output" / "mars_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"L2 distance from Earth:   {R_L2_EARTH/1e6:.3f} million km")
    print(f"L2 parking speed:         {V_L2:.2f} m/s")
    print(f"Earth orbital speed:      {np.sqrt(G*SUN_MASS/R_EARTH_SUN):.2f} m/s")
    print(f"Hohmann delta-v L2→Mars:  {HOHMANN_DV:.1f} m/s")
    print(f"\nSweep: {len(launch_speeds)} x {len(angles)} = "
          f"{len(launch_speeds)*len(angles)} runs")
    print(f"{'dv (m/s)':>10} {'Angle':>8} {'Min dist (AU)':>14} "
          f"{'Days':>8} {'Returns?':>9} {'Fuel (kg)':>10}")
    print("-" * 65)

    for dv in launch_speeds:
        for theta in angles:
            sim = Simulation(dt=dt, integrator="beeman")
            sim.load_bodies_from_json(data_file)
            sim.initialise_bodies(sun_mass=SUN_MASS)

            satellite = add_satellite(sim, dv, theta)
            mars      = next(b for b in sim.bodies if b.name == "Mars")
            earth     = next(b for b in sim.bodies if b.name == "Earth")

            min_dist_m     = np.inf
            journey_time_s = 0.0
            returned       = False
            traj           = []

            for i in range(total_steps):
                sim.step()
                sat_pos    = satellite.position.copy()
                dist_mars  = np.linalg.norm(sat_pos - mars.position)
                dist_earth = np.linalg.norm(sat_pos - earth.position)
                traj.append(sat_pos / AU)

                if sim.time > 60 * DAY and dist_mars < min_dist_m:
                    min_dist_m     = dist_mars
                    journey_time_s = sim.time

                if journey_time_s > 0 and sim.time > journey_time_s + 30 * DAY:
                    a_sat   = np.arctan2(sat_pos[1], sat_pos[0])
                    a_earth = np.arctan2(earth.position[1], earth.position[0])
                    d_ang   = abs((a_sat - a_earth + np.pi) % (2*np.pi) - np.pi)
                    if (dist_earth / AU < EARTH_RETURN_THRESHOLD_AU
                            and np.degrees(d_ang) < EARTH_RETURN_ANGLE_DEG):
                        returned = True

            min_dist_au  = min_dist_m / AU
            journey_days = journey_time_s / DAY
            fuel_kg      = fuel_mass(dv)

            print(f"{dv:>10.1f} {theta:>8.1f} {min_dist_au:>14.4f} "
                  f"{journey_days:>8.1f} {'Yes' if returned else 'No':>9} "
                  f"{fuel_kg:>10.1f}")

            rows.append({
                "delta_v_ms":   dv,
                "angle_deg":    theta,
                "min_dist_au":  min_dist_au,
                "journey_days": journey_days,
                "returned":     returned,
                "fuel_kg":      fuel_kg,
                "traj":         traj,
                "sim":          sim,
            })

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("traj", "sim")}
                       for r in rows])
    df.to_csv(output_dir / "parameter_sweep.csv", index=False)
    print(f"\nCSV saved → {output_dir / 'parameter_sweep.csv'}")

    returning = [r for r in rows if r["returned"]]
    if not returning:
        print("\nNo trajectories returned. Selecting best overall.")
        returning = rows

    best = select_best(returning)

    print(f"\nBest trajectory:")
    print(f"  Delta-v:     {best['delta_v_ms']:.1f} m/s  (Hohmann ref: {HOHMANN_DV:.1f} m/s)")
    print(f"  Angle:       {best['angle_deg']:.1f} deg from prograde")
    print(f"  Mars dist:   {best['min_dist_au']:.4f} AU  ({best['min_dist_au']*AU/1e6:.0f} km)")
    print(f"  Journey:     {best['journey_days']:.1f} days  (Perseverance: {PERSEVERANCE_JOURNEY_DAYS:.0f})")
    print(f"  Fuel:        {best['fuel_kg']:.1f} kg  (wet mass: {PAYLOAD_MASS+best['fuel_kg']:.1f} kg)")
    print(f"  Score:       {best['score']:.4f}  (dist=0.5, time=0.3, fuel=0.2)")

    plot_trajectory(best, dt)
    plot_time_vs_fuel(returning, best)
    animate_trajectory(data_file, best["delta_v_ms"], best["angle_deg"],
                        dt, best["journey_days"], best["min_dist_au"])


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_trajectory(best: dict, dt: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    sim   = best["sim"]
    t_arr = np.linspace(0, 2 * np.pi, 300)

    for body in sim.bodies:
        if body.name in ("Sun", "Satellite"):
            continue
        r_au = body.orbital_radius / AU
        ax.plot(r_au * np.cos(t_arr), r_au * np.sin(t_arr),
                color=body.colour, linewidth=0.4, alpha=0.3)
        ax.plot(*body.position / AU, "o", color=body.colour, markersize=4)
        ax.text(body.position[0]/AU, body.position[1]/AU + 0.06,
                body.name, color=body.colour, fontsize=6, ha="center")

    ax.plot(0, 0, "o", color="yellow", markersize=10, zorder=5)

    earth      = next(b for b in sim.bodies if b.name == "Earth")
    earth_rhat = earth.position / np.linalg.norm(earth.position)
    l2_pos     = (earth.position - earth_rhat * R_L2_EARTH) / AU
    ax.plot(*l2_pos, "D", color="cyan", markersize=7, zorder=6, label="L2 (launch)")

    traj = np.array(best["traj"])
    ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=0.8,
            label=f"Δv={best['delta_v_ms']:.0f} m/s, θ={best['angle_deg']:.0f}°")

    cs = min(int(best["journey_days"] * DAY / dt), len(traj) - 1)
    ax.plot(traj[cs, 0], traj[cs, 1], "*", color="red", markersize=12,
            label=f"Closest ({best['min_dist_au']:.3f} AU)")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
    ax.legend(fontsize=8, facecolor="#111111", labelcolor="white")
    ax.set_title("Experiment 3: L2 → Mars (best trajectory)", color="white")
    plt.tight_layout()
    plt.show()


def plot_time_vs_fuel(returning: list[dict], best: dict) -> None:
    times  = np.array([r["journey_days"] for r in returning])
    fuels  = np.array([r["fuel_kg"]      for r in returning])
    angles = np.array([r["angle_deg"]    for r in returning])
    dists  = np.array([r["min_dist_au"]  for r in returning])

    # constrained_layout handles colorbars correctly — tight_layout struggles with them
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sc = axes[0].scatter(fuels, times, c=angles, cmap="plasma", s=60, alpha=0.8)
    plt.colorbar(sc, ax=axes[0], label="Burn angle (deg)")
    axes[0].scatter(best["fuel_kg"], best["journey_days"],
                    marker="*", s=250, color="red", zorder=5,
                    label=f"Best  Δv={best['delta_v_ms']:.0f}, θ={best['angle_deg']:.0f}°")
    axes[0].axhline(PERSEVERANCE_JOURNEY_DAYS, color="cyan", linewidth=1.0,
                    linestyle="--", label=f"Perseverance ({PERSEVERANCE_JOURNEY_DAYS:.0f} d)")
    axes[0].set_xlabel("Fuel mass (kg)")
    axes[0].set_ylabel("Journey time (days)")
    axes[0].set_title("Journey time vs Fuel (returning trajectories)")
    axes[0].legend(fontsize=8)

    sc2 = axes[1].scatter(fuels, dists, c=angles, cmap="plasma", s=60, alpha=0.8)
    plt.colorbar(sc2, ax=axes[1], label="Burn angle (deg)")
    axes[1].scatter(best["fuel_kg"], best["min_dist_au"],
                    marker="*", s=250, color="red", zorder=5, label="Best")
    axes[1].set_xlabel("Fuel mass (kg)")
    axes[1].set_ylabel("Closest approach to Mars (AU)")
    axes[1].set_title("Closest approach vs Fuel (returning trajectories)")
    axes[1].legend(fontsize=8)

    plt.suptitle("Experiment 3: L2 Launch — Trade-off Analysis", fontsize=12)
    plt.show()


def animate_trajectory(data_file: str, delta_v: float, theta_deg: float,
                         dt: float, journey_days: float, min_dist_au: float) -> None:
    MAX_YEARS   = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    sim = Simulation(dt=dt, integrator="beeman")
    sim.load_bodies_from_json(data_file)
    sim.initialise_bodies(sun_mass=SUN_MASS)
    satellite = add_satellite(sim, delta_v, theta_deg)

    animated_names = [b.name for b in sim.bodies if b.name != "Sun"]
    body_colour    = {b.name: b.colour for b in sim.bodies}
    body_orbit_au  = {b.name: b.orbital_radius / AU
                      for b in sim.bodies if b.name not in ("Sun", "Satellite")}
    body_map       = {b.name: b for b in sim.bodies}

    history    = {name: np.empty((total_steps, 2)) for name in animated_names}
    times_days = np.empty(total_steps)
    for step in range(total_steps):
        sim.step()
        for name in animated_names:
            history[name][step] = body_map[name].position / AU
        times_days[step] = sim.time / DAY

    n_frames  = min(800, total_steps)
    frame_idx = np.linspace(0, total_steps - 1, n_frames, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
    ax.set_title(f"L2 → Mars  Δv={delta_v:.0f} m/s, θ={theta_deg:.0f}°", color="white")

    ring_t = np.linspace(0, 2 * np.pi, 300)
    for name, r_au in body_orbit_au.items():
        ax.plot(r_au * np.cos(ring_t), r_au * np.sin(ring_t),
                color=body_colour[name], linewidth=0.4, alpha=0.3)

    ax.plot(0, 0, "o", color="yellow", markersize=10, zorder=5)
    ax.plot(history["Satellite"][0, 0], history["Satellite"][0, 1],
            "D", color="cyan", markersize=7, label="L2 launch", zorder=6)

    sat_trail, = ax.plot([], [], color="white", linewidth=0.9, alpha=0.9, zorder=3)
    sat_dot,   = ax.plot([], [], "o", color="cyan", markersize=6, zorder=4)
    planet_dots, planet_labels = {}, {}
    for name in animated_names:
        if name == "Satellite":
            continue
        dot, = ax.plot([], [], "o", color=body_colour.get(name, "grey"),
                       markersize=5, zorder=4)
        lbl  = ax.text(0, 0, name, color=body_colour.get(name, "grey"),
                       fontsize=6, ha="center", zorder=4)
        planet_dots[name]   = dot
        planet_labels[name] = lbl

    closest_marker, = ax.plot([], [], "*", color="red", markersize=14, zorder=6,
                               label=f"Closest ({min_dist_au:.3f} AU)")
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        color="white", fontsize=9, va="top")
    ax.legend(fontsize=8, facecolor="#111111", labelcolor="white", loc="upper right")

    closest_global = int(journey_days * DAY / dt)
    closest_frame  = int(np.argmin(np.abs(frame_idx - closest_global)))

    def init():
        sat_trail.set_data([], [])
        sat_dot.set_data([], [])
        closest_marker.set_data([], [])
        time_text.set_text("")
        for dot in planet_dots.values():
            dot.set_data([], [])
        return ([sat_trail, sat_dot, closest_marker, time_text]
                + list(planet_dots.values()))

    def update(fn):
        idx = frame_idx[fn]
        sat_trail.set_data(history["Satellite"][:idx+1, 0],
                           history["Satellite"][:idx+1, 1])
        sat_dot.set_data([history["Satellite"][idx, 0]],
                         [history["Satellite"][idx, 1]])
        for name, dot in planet_dots.items():
            pos = history[name][idx]
            dot.set_data([pos[0]], [pos[1]])
            planet_labels[name].set_position((pos[0], pos[1] + 0.06))
        if fn >= closest_frame:
            ca = history["Satellite"][frame_idx[closest_frame]]
            closest_marker.set_data([ca[0]], [ca[1]])
        time_text.set_text(f"t = {times_days[idx]:.0f} days")
        return ([sat_trail, sat_dot, closest_marker, time_text]
                + list(planet_dots.values()))

    # Must store in a variable — if not assigned, Python's garbage collector
    # destroys the FuncAnimation object immediately and the animation never runs
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=20, blit=True)
    plt.tight_layout()
    plt.show()