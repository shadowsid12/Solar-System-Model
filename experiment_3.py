"""
experiment_3.py
---------------
Experiment 3: Satellite to Mars

Searches over launch speeds and launch angles to find trajectories that
bring a satellite close to Mars. Reports closest approach, journey time
vs Perseverance, return-to-Earth status, and fuel via the rocket equation.

Parameter sweep
---------------
- extra_speed : additional speed (m/s) added on top of Earth's orbital velocity
- theta       : angle (degrees) of the extra velocity vector measured from
                Earth's velocity direction (theta=0 is purely tangential,
                theta=90 is purely radial outward, etc.)

Outputs
-------
- CSV of all sweep results (output/mars_experiment/parameter_sweep.csv)
- Static trajectory plot for the best returning trajectory
- Scatter plot: journey time vs fuel for all returning trajectories
- Animation of the best returning trajectory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from simulation import Simulation
from bodies import Body

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

AU         = 1.496e11             # metres per astronomical unit
DAY        = 24 * 3600            # seconds per day
EARTH_YEAR = 365.25 * 24 * 3600  # seconds per Earth year
SUN_MASS   = 1.989e30             # kg

PERSEVERANCE_JOURNEY_DAYS = 203.0  # Jul 30 2020 → Feb 18 2021

PAYLOAD_MASS = 2000.0   # kg
V_EXHAUST    = 4400.0   # m/s  (~Isp 450 s for chemical propulsion)

# Return-to-Earth detection thresholds.
# Both conditions must be satisfied simultaneously:
#   1. Distance to Earth < EARTH_RETURN_THRESHOLD_AU
#      (~0.01 AU = ~1.5 million km, roughly Earth's Hill sphere radius)
#   2. Angular separation from Earth < EARTH_RETURN_ANGLE_DEG
#      Prevents false positives where the satellite is at ~1 AU from the Sun
#      but on the opposite side — Earth and satellite at similar radii but
#      different angular positions would otherwise both pass the distance check
#      once Earth completes an orbit and laps the satellite.
EARTH_RETURN_THRESHOLD_AU  = 0.01   # AU — radial proximity to Earth
EARTH_RETURN_ANGLE_DEG     = 5.0    # degrees — angular proximity to Earth
LAUNCH_OFFSET_M            = 1.0e9  # metres from Earth — sphere of influence boundary


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def fuel_mass(delta_v: float) -> float:
    """
    Tsiolkovsky rocket equation.
    Returns fuel mass (kg) needed to achieve delta_v with a fixed payload.
        m_fuel = m_payload * (exp(dv / v_exhaust) - 1)
    """
    return PAYLOAD_MASS * (np.exp(delta_v / V_EXHAUST) - 1)


def add_satellite(sim: Simulation, extra_speed: float, theta_deg: float) -> Body:
    """
    Add a satellite to the simulation launched from near Earth.

    The satellite is placed LAUNCH_OFFSET_M radially outward from Earth
    (away from the Sun), so Earth's gravity is negligible at launch.

    The extra velocity is applied at angle theta_deg relative to Earth's
    velocity vector:
        theta=0   → purely tangential (prograde, along Earth's orbit)
        theta=90  → purely radial outward (away from Sun)
        theta=180 → retrograde (opposing Earth's orbit)

    Parameters
    ----------
    sim         : Simulation (planets already initialised)
    extra_speed : magnitude of extra velocity (m/s)
    theta_deg   : direction of extra velocity relative to Earth's velocity (degrees)
    """
    earth = next(b for b in sim.bodies if b.name == "Earth")

    # Radial unit vector: Sun → Earth
    earth_rhat = earth.position / np.linalg.norm(earth.position)

    # Tangential unit vector: direction of Earth's orbital motion
    earth_vhat = earth.velocity / np.linalg.norm(earth.velocity)

    # Rotate extra_speed direction by theta relative to Earth's velocity
    theta_rad  = np.radians(theta_deg)
    extra_vhat = np.cos(theta_rad) * earth_vhat + np.sin(theta_rad) * earth_rhat

    satellite = Body(
        name="Satellite",
        mass=PAYLOAD_MASS,
        orbital_radius=0.0,
        colour="white",
        is_satellite=True,
    )
    satellite.position = earth.position + earth_rhat * LAUNCH_OFFSET_M
    satellite.velocity = earth.velocity + extra_speed * extra_vhat

    sim.add_body(satellite)
    accels = sim.compute_accelerations()
    satellite.acceleration      = accels["Satellite"]
    satellite.prev_acceleration = accels["Satellite"].copy()

    return satellite


# ------------------------------------------------------------------
# Main experiment function
# ------------------------------------------------------------------

def run_experiment_3(data_file: str, launch_speeds: list[float],
                     angles: list[float], dt: float) -> None:
    """
    Sweep over launch speeds and angles. For each combination, runs a fresh
    simulation for MAX_YEARS and records:
        - minimum distance to Mars (AU)
        - journey time to closest approach (days)
        - whether the satellite returns within EARTH_RETURN_THRESHOLD_AU of Earth
        - fuel mass required (kg)

    Saves results to CSV, then produces:
        1. Static trajectory plot of the best returning trajectory
        2. Scatter plot: journey time vs fuel for all returning trajectories
        3. Animation of the best returning trajectory

    Parameters
    ----------
    data_file    : path to planets.json
    launch_speeds: list of extra speeds (m/s) above Earth's orbital velocity
    angles       : list of launch angles in degrees (0 = prograde tangential)
    dt           : time step in seconds
    """
    MAX_YEARS   = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    output_dir = Path(__file__).parent / "output" / "mars_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Sweep: {len(launch_speeds)} speeds x {len(angles)} angles "
          f"= {len(launch_speeds)*len(angles)} runs")
    print(f"{'Speed (m/s)':>12} {'Angle (deg)':>12} {'Min dist (AU)':>14} "
          f"{'Journey (days)':>15} {'Returns?':>9} {'Fuel (kg)':>10}")
    print("-" * 76)

    for extra_v in launch_speeds:
        for theta in angles:
            sim = Simulation(dt=dt, integrator="beeman")
            sim.load_bodies_from_json(data_file)
            sim.initialise_bodies(sun_mass=SUN_MASS)

            satellite = add_satellite(sim, extra_v, theta)
            mars      = next(b for b in sim.bodies if b.name == "Mars")
            earth     = next(b for b in sim.bodies if b.name == "Earth")

            min_dist_m     = np.inf
            journey_time_s = 0.0
            returned       = False
            traj           = []   # list of (x_AU, y_AU) positions

            for _ in range(total_steps):
                sim.step()

                sat_pos = satellite.position.copy()
                traj.append(sat_pos / AU)

                dist_mars  = np.linalg.norm(sat_pos - mars.position)
                dist_earth = np.linalg.norm(sat_pos - earth.position)

                # Only track Mars closest approach after 60 days
                if sim.time > 60 * DAY and dist_mars < min_dist_m:
                    min_dist_m     = dist_mars
                    journey_time_s = sim.time

                # Return check — requires BOTH conditions after the Mars fly-past:
                #   1. Within EARTH_RETURN_THRESHOLD_AU of Earth's actual position
                #   2. Within EARTH_RETURN_ANGLE_DEG of Earth's angular position
                # Condition 2 is critical: without it, any trajectory that passes
                # through ~1 AU after ~1 year will appear to "return" because Earth
                # has orbited back to a similar radius, making the distance check pass
                # even when they are on opposite sides of the Sun.
                if journey_time_s > 0 and sim.time > journey_time_s + 30 * DAY:
                    angle_sat   = np.arctan2(sat_pos[1], sat_pos[0])
                    angle_earth = np.arctan2(earth.position[1], earth.position[0])
                    delta_angle = abs((angle_sat - angle_earth + np.pi) % (2*np.pi) - np.pi)
                    delta_angle_deg = np.degrees(delta_angle)
                    if (dist_earth / AU < EARTH_RETURN_THRESHOLD_AU
                            and delta_angle_deg < EARTH_RETURN_ANGLE_DEG):
                        returned = True

            min_dist_au  = min_dist_m / AU
            journey_days = journey_time_s / DAY
            fuel_kg      = fuel_mass(extra_v)

            print(f"{extra_v:>12.1f} {theta:>12.1f} {min_dist_au:>14.4f} "
                  f"{journey_days:>15.1f} {'Yes' if returned else 'No':>9} "
                  f"{fuel_kg:>10.1f}")

            rows.append({
                "extra_v_ms":    extra_v,
                "angle_deg":     theta,
                "min_dist_au":   min_dist_au,
                "journey_days":  journey_days,
                "returned":      returned,
                "fuel_kg":       fuel_kg,
                "traj":          traj,   # not saved to CSV
                "sim":           sim,    # not saved to CSV
            })

    # --- Save CSV (exclude non-serialisable columns) ---
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("traj", "sim")}
                       for r in rows])
    csv_path = output_dir / "parameter_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # --- Find best returning trajectory using weighted score ---
    returning = [r for r in rows if r["returned"]]
    if not returning:
        print("\nNo trajectories returned to Earth. Showing best overall.")
        returning = rows

    best = _select_best(returning)

    print(f"\nBest returning trajectory (weighted score):")
    print(f"  Speed:                    {best['extra_v_ms']:.1f} m/s")
    print(f"  Angle:                    {best['angle_deg']:.1f} deg")
    print(f"  Closest approach to Mars: {best['min_dist_au']:.4f} AU  "
          f"({best['min_dist_au']*AU/1e6:.0f} km)")
    print(f"  Journey time:             {best['journey_days']:.1f} days  "
          f"(Perseverance: {PERSEVERANCE_JOURNEY_DAYS:.0f} days)")
    print(f"  Fuel required:            {best['fuel_kg']:.1f} kg  "
          f"(dry mass: {PAYLOAD_MASS:.0f} kg, wet mass: {PAYLOAD_MASS + best['fuel_kg']:.1f} kg)")
    print(f"  Weighted score:           {best.get('score', float('nan')):.4f}  "
          f"(lower = better; weights: dist=0.5, time=0.3, fuel=0.2)")

    # --- Plot 1: static trajectory of best run ---
    _plot_best_trajectory(best, dt)

    # --- Plot 2: scatter — journey time vs fuel for all returning trajectories ---
    _plot_time_vs_fuel(returning, best)

    # --- Animation: best trajectory ---
    _animate_trajectory(data_file, best["extra_v_ms"], best["angle_deg"], dt,
                        best["journey_days"], best["min_dist_au"])


# ------------------------------------------------------------------
# Visualisation helpers
# ------------------------------------------------------------------

def _select_best(candidates: list[dict]) -> dict:
    """
    Select the best trajectory from a list using a weighted normalised score.

    Each metric is normalised to [0, 1] across all candidates so that
    differences in units (AU vs days vs kg) do not bias the result.
    Lower score = better trajectory.

    Weights (sum to 1.0):
        0.5 — closest Mars approach distance
              Weighted most heavily because the primary mission goal is a
              fly-past; a trajectory that barely reaches Mars orbit is not useful
              regardless of how cheap or fast it is.
        0.3 — journey time to closest approach
              Shorter missions reduce operational risk, communication delays,
              and time for systems to degrade.
        0.2 — fuel mass required
              Lower fuel means lower launch cost and simpler vehicle design,
              but fuel is already partially captured by time (faster = more fuel)
              so it is down-weighted relative to the other two.

    Score formula:
        score = 0.5 * (d / d_max) + 0.3 * (t / t_max) + 0.2 * (f / f_max)

    where d = min_dist_au, t = journey_days, f = fuel_kg, and the denominators
    are the maximum values across all candidates (normalisation).
    """
    d_max = max(r["min_dist_au"]  for r in candidates)
    t_max = max(r["journey_days"] for r in candidates)
    f_max = max(r["fuel_kg"]      for r in candidates)

    def score(r: dict) -> float:
        d_norm = r["min_dist_au"]  / d_max if d_max > 0 else 0
        t_norm = r["journey_days"] / t_max if t_max > 0 else 0
        f_norm = r["fuel_kg"]      / f_max if f_max > 0 else 0
        return 0.5 * d_norm + 0.3 * t_norm + 0.2 * f_norm

    best = min(candidates, key=score)
    best["score"] = score(best)
    return best


def _plot_best_trajectory(best: dict, dt: float) -> None:
    """Static trajectory plot for the best returning run."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    sim   = best["sim"]
    theta = np.linspace(0, 2 * np.pi, 300)

    # Faint orbit rings + final planet positions
    for body in sim.bodies:
        if body.name in ("Sun", "Satellite"):
            continue
        r_au = body.orbital_radius / AU
        ax.plot(r_au * np.cos(theta), r_au * np.sin(theta),
                color=body.colour, linewidth=0.4, alpha=0.3)
        ax.plot(*body.position / AU, "o", color=body.colour, markersize=4)
        ax.text(body.position[0]/AU, body.position[1]/AU + 0.06,
                body.name, color=body.colour, fontsize=6, ha="center")

    ax.plot(0, 0, "o", color="yellow", markersize=10, zorder=5)

    # Satellite trajectory
    traj = np.array(best["traj"])
    ax.plot(traj[:, 0], traj[:, 1], color="white", linewidth=0.8, alpha=0.9,
            label=f"Satellite  Δv={best['extra_v_ms']:.0f} m/s, θ={best['angle_deg']:.0f}°")
    ax.plot(traj[0, 0], traj[0, 1], "^", color="lime", markersize=8, label="Launch")

    # Closest-approach marker
    closest_step = min(int(best["journey_days"] * DAY / (EARTH_YEAR/1000)),
                       len(traj) - 1)
    ax.plot(traj[closest_step, 0], traj[closest_step, 1], "*", color="red",
            markersize=12, label=f"Closest ({best['min_dist_au']:.3f} AU)")

    lim = 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(fontsize=8, facecolor="#111111", labelcolor="white")
    ax.set_title("Experiment 3: Best Returning Trajectory to Mars", color="white")
    plt.tight_layout()
    plt.show()


def _plot_time_vs_fuel(returning: list[dict], best: dict) -> None:
    """
    Scatter plot: journey time (days) vs fuel mass (kg) for all returning
    trajectories. Each point is coloured by launch angle. The best trajectory
    is highlighted.
    """
    speeds  = np.array([r["extra_v_ms"]   for r in returning])
    times   = np.array([r["journey_days"] for r in returning])
    fuels   = np.array([r["fuel_kg"]      for r in returning])
    angles  = np.array([r["angle_deg"]    for r in returning])
    dists   = np.array([r["min_dist_au"]  for r in returning])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: journey time vs fuel, coloured by angle ---
    sc = axes[0].scatter(fuels, times, c=angles, cmap="plasma",
                         s=60, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=axes[0], label="Launch angle (deg)")

    # Highlight best
    axes[0].scatter(best["fuel_kg"], best["journey_days"],
                    marker="*", s=250, color="red", zorder=5,
                    label=f"Best  Δv={best['extra_v_ms']:.0f} m/s, θ={best['angle_deg']:.0f}°")

    # Perseverance reference line
    axes[0].axhline(PERSEVERANCE_JOURNEY_DAYS, color="cyan", linewidth=1.0,
                    linestyle="--", label=f"Perseverance ({PERSEVERANCE_JOURNEY_DAYS:.0f} days)")

    axes[0].set_xlabel("Fuel mass (kg)")
    axes[0].set_ylabel("Journey time to closest approach (days)")
    axes[0].set_title("Returning trajectories: Journey time vs Fuel")
    axes[0].legend(fontsize=8)

    # --- Right: closest approach distance vs fuel, coloured by angle ---
    sc2 = axes[1].scatter(fuels, dists, c=angles, cmap="plasma",
                          s=60, alpha=0.8, zorder=3)
    plt.colorbar(sc2, ax=axes[1], label="Launch angle (deg)")

    axes[1].scatter(best["fuel_kg"], best["min_dist_au"],
                    marker="*", s=250, color="red", zorder=5, label="Best")

    axes[1].set_xlabel("Fuel mass (kg)")
    axes[1].set_ylabel("Closest approach to Mars (AU)")
    axes[1].set_title("Returning trajectories: Closest approach vs Fuel")
    axes[1].legend(fontsize=8)

    plt.suptitle("Experiment 3: Trade-off Analysis — Returning Trajectories", fontsize=12)
    plt.tight_layout()
    plt.show()


def _animate_trajectory(data_file: str, extra_v: float, theta_deg: float,
                         dt: float, journey_days: float, min_dist_au: float) -> None:
    """
    Re-run the best trajectory and animate the satellite's journey.
    Shows planet orbits, live planet positions, growing satellite trail,
    and reveals a closest-approach marker when the satellite arrives.
    """
    MAX_YEARS   = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    # Re-run to collect full position history
    sim = Simulation(dt=dt, integrator="beeman")
    sim.load_bodies_from_json(data_file)
    sim.initialise_bodies(sun_mass=SUN_MASS)
    satellite = add_satellite(sim, extra_v, theta_deg)

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

    # Subsample to at most 800 frames
    n_frames  = min(800, total_steps)
    frame_idx = np.linspace(0, total_steps - 1, n_frames, dtype=int)

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    lim = 2.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.set_title(f"Experiment 3: Satellite to Mars  "
                 f"(Δv={extra_v:.0f} m/s, θ={theta_deg:.0f}°)", color="white")

    # Static: orbit rings + Sun + launch marker
    ring_theta = np.linspace(0, 2 * np.pi, 300)
    for name, r_au in body_orbit_au.items():
        ax.plot(r_au * np.cos(ring_theta), r_au * np.sin(ring_theta),
                color=body_colour[name], linewidth=0.4, alpha=0.3)

    ax.plot(0, 0, "o", color="yellow", markersize=10, zorder=5)
    ax.plot(history["Satellite"][0, 0], history["Satellite"][0, 1],
            "^", color="lime", markersize=7, label="Launch", zorder=5)

    # Animated artists
    sat_trail, = ax.plot([], [], color="white", linewidth=0.9, alpha=0.9, zorder=3)
    sat_dot,   = ax.plot([], [], "o", color="cyan", markersize=6, zorder=4)

    planet_dots   = {}
    planet_labels = {}
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

    # Frame at which closest approach occurs
    closest_global_step = int(journey_days * DAY / dt)
    closest_frame       = int(np.argmin(np.abs(frame_idx - closest_global_step)))

    def init():
        sat_trail.set_data([], [])
        sat_dot.set_data([], [])
        closest_marker.set_data([], [])
        time_text.set_text("")
        for dot in planet_dots.values():
            dot.set_data([], [])
        return ([sat_trail, sat_dot, closest_marker, time_text]
                + list(planet_dots.values()))

    def update(frame_num):
        idx = frame_idx[frame_num]

        sat_trail.set_data(history["Satellite"][:idx+1, 0],
                           history["Satellite"][:idx+1, 1])
        sat_dot.set_data([history["Satellite"][idx, 0]],
                         [history["Satellite"][idx, 1]])

        for name, dot in planet_dots.items():
            pos = history[name][idx]
            dot.set_data([pos[0]], [pos[1]])
            planet_labels[name].set_position((pos[0], pos[1] + 0.06))

        if frame_num >= closest_frame:
            ca = history["Satellite"][frame_idx[closest_frame]]
            closest_marker.set_data([ca[0]], [ca[1]])

        time_text.set_text(f"t = {times_days[idx]:.0f} days")

        return ([sat_trail, sat_dot, closest_marker, time_text]
                + list(planet_dots.values()))

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=20, blit=True)
    plt.tight_layout()
    plt.show()