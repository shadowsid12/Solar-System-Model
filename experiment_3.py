"""
experiment_3.py
---------------
Experiment 3: Satellite to Mars

Searches for launch velocities that send a satellite close to Mars.
Reports closest approach, journey time vs Perseverance, return-to-Earth
status, and fuel required via the rocket equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simulation import Simulation
from bodies import Body

AU         = 1.496e11            # metres per astronomical unit
EARTH_YEAR = 365.25 * 24 * 3600
SUN_MASS   = 1.989e30

# Perseverance reference journey time: launched July 30 2020, landed Feb 18 2021
PERSEVERANCE_JOURNEY_DAYS = 203.0

# Rocket equation constants (chemical propulsion, e.g. Atlas V upper stage)
PAYLOAD_MASS = 2000.0   # kg — fixed satellite payload mass
V_EXHAUST    = 4400.0   # m/s — exhaust velocity (~Isp 450 s * 9.81)

# "Return to Earth" threshold: satellite comes within this distance of Earth
EARTH_RETURN_THRESHOLD_AU = 0.1   # AU


def fuel_mass(delta_v: float) -> float:
    """
    Rocket equation: m_fuel = m_payload * (exp(dv / v_exhaust) - 1)
    Returns fuel mass in kg required to achieve delta_v from rest relative
    to Earth's orbit, given a fixed payload mass.
    """
    return PAYLOAD_MASS * (np.exp(delta_v / V_EXHAUST) - 1)


def add_satellite(sim: Simulation, extra_speed: float) -> Body:
    """
    Create and add a satellite launched from just outside Earth's position.

    The satellite starts 1 million km above Earth (radially outward from the
    Sun), inherits Earth's full velocity, and receives extra_speed added
    tangentially (in the direction of Earth's velocity unit vector).

    Parameters
    ----------
    sim         : Simulation to add the satellite to
    extra_speed : additional speed in m/s added tangent to Earth's orbit
    """
    earth = next(b for b in sim.bodies if b.name == "Earth")

    # Launch from Earth's sphere of influence boundary (~1e9 m, ~2.5x Moon distance).
    # At this distance Earth's gravity is negligible vs the Sun's, so the satellite
    # behaves as a heliocentric particle from the start.
    # The offset is radial (away from Sun) to avoid numerical issues.
    earth_rhat = earth.position / np.linalg.norm(earth.position)
    offset     = earth_rhat * 1.0e9   # 1 million km radially outward

    # Launch tangentially — add extra_speed in the direction Earth is moving
    earth_vhat = earth.velocity / np.linalg.norm(earth.velocity)
    launch_vel = earth.velocity + extra_speed * earth_vhat

    satellite = Body(
        name="Satellite",
        mass=PAYLOAD_MASS,
        orbital_radius=0.0,   # not used for satellites
        colour="white",
        is_satellite=True,
    )
    satellite.position = earth.position + offset
    satellite.velocity = launch_vel

    # Bootstrap acceleration so Beeman has a valid a(t-dt) at t=0
    sim.add_body(satellite)
    accels = sim.compute_accelerations()
    satellite.acceleration      = accels["Satellite"]
    satellite.prev_acceleration = accels["Satellite"].copy()

    return satellite


def run_experiment_3(data_file: str, launch_speeds: list[float], dt: float) -> None:
    """
    Experiment 3: Satellite to Mars
    --------------------------------
    For each extra launch speed, runs a fresh simulation and tracks:
      - Minimum distance to Mars
      - Journey time to the closest approach (days)
      - Whether the satellite returns within EARTH_RETURN_THRESHOLD_AU of Earth
      - Fuel mass required (rocket equation)

    Produces:
      - A printed results table for all launch speeds
      - A static trajectory plot for the best fly-past
      - A bar chart of fuel mass vs launch speed
      - An animation of the best trajectory (via anim_best_trajectory)

    Parameters
    ----------
    data_file    : path to planets.json
    launch_speeds: list of extra speeds (m/s) above Earth's orbital velocity
    dt           : time step in seconds
    """
    DAY       = 24 * 3600
    MAX_YEARS = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    results = []

    header = (f"{'Extra v (m/s)':>14} {'Min dist (AU)':>14} "
              f"{'Journey (days)':>15} {'Returns?':>9} {'Fuel (kg)':>10}")
    print(header)
    print("-" * 66)

    best_idx  = None
    best_dist = np.inf

    for extra_v in launch_speeds:
        sim = Simulation(dt=dt, integrator="beeman")
        sim.load_bodies_from_json(data_file)
        sim.initialise_bodies(sun_mass=SUN_MASS)
        satellite = add_satellite(sim, extra_v)

        mars  = next(b for b in sim.bodies if b.name == "Mars")
        earth = next(b for b in sim.bodies if b.name == "Earth")

        min_dist_m     = np.inf
        journey_time_s = None
        returned       = False
        traj_positions = []   # satellite (x, y) positions in metres

        for step in range(total_steps):
            sim.step()

            sat_pos  = satellite.position
            dist_m   = np.linalg.norm(sat_pos - mars.position)

            traj_positions.append(sat_pos.copy())

            # Only track closest approach after 60 days — lets the satellite
            # clearly depart Earth's vicinity before logging Mars proximity
            if sim.time > 60 * DAY and dist_m < min_dist_m:
                min_dist_m     = dist_m
                journey_time_s = sim.time

            # Check return to Earth (after at least 30 days to avoid false trigger)
            dist_to_earth = np.linalg.norm(sat_pos - earth.position)
            if sim.time > 30 * DAY and dist_to_earth / AU < EARTH_RETURN_THRESHOLD_AU:
                returned = True

        fuel_kg      = fuel_mass(extra_v)
        min_dist_au  = min_dist_m / AU
        journey_days = journey_time_s / DAY

        results.append({
            "extra_v":      extra_v,
            "min_dist_au":  min_dist_au,
            "journey_days": journey_days,
            "returned":     returned,
            "fuel_kg":      fuel_kg,
            "traj":         traj_positions,
            "sim":          sim,
        })

        print(f"{extra_v:>14.1f} {min_dist_au:>14.4f} {journey_days:>15.1f} "
              f"{'Yes' if returned else 'No':>9} {fuel_kg:>10.1f}")

        if min_dist_au < best_dist:
            best_dist = min_dist_au
            best_idx  = len(results) - 1

    # --- Summary ---
    best = results[best_idx]
    print(f"\nBest trajectory: extra_v = {best['extra_v']:.1f} m/s")
    print(f"  Closest approach to Mars : {best['min_dist_au']:.4f} AU  "
          f"({best['min_dist_au'] * AU / 1e6:.0f} km)")
    print(f"  Journey time             : {best['journey_days']:.1f} days  "
          f"(Perseverance: {PERSEVERANCE_JOURNEY_DAYS:.0f} days)")
    print(f"  Returns to Earth         : {'Yes' if best['returned'] else 'No'}")
    print(f"  Fuel required            : {best['fuel_kg']:.1f} kg  "
          f"(total launch mass: {PAYLOAD_MASS + best['fuel_kg']:.1f} kg)")

    # --- Plot 1: best trajectory (static) ---
    fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor="black")
    ax1.set_facecolor("black")
    ax1.set_aspect("equal")

    best_sim = best["sim"]
    theta    = np.linspace(0, 2 * np.pi, 300)

    for body in best_sim.bodies:
        if body.name in ("Sun", "Satellite"):
            continue
        r_au = body.orbital_radius / AU
        ax1.plot(r_au * np.cos(theta), r_au * np.sin(theta),
                 color=body.colour, linewidth=0.4, alpha=0.3)
        ax1.plot(*body.position / AU, "o", color=body.colour, markersize=4)
        ax1.text(body.position[0] / AU, body.position[1] / AU + 0.08,
                 body.name, color=body.colour, fontsize=6, ha="center")

    ax1.plot(0, 0, "o", color="yellow", markersize=10)

    traj = np.array(best["traj"]) / AU
    ax1.plot(traj[:, 0], traj[:, 1], color="white", linewidth=0.8,
             label=f"Satellite (Δv = {best['extra_v']:.0f} m/s)")
    ax1.plot(traj[0, 0], traj[0, 1], "^", color="lime", markersize=7, label="Launch")

    closest_step = int(best["journey_days"] * DAY / dt)
    closest_step = min(closest_step, len(traj) - 1)
    ax1.plot(traj[closest_step, 0], traj[closest_step, 1],
             "*", color="red", markersize=10,
             label=f"Closest ({best['min_dist_au']:.3f} AU)")

    MARS_ORBIT_AU = 1.524
    plot_limit    = MARS_ORBIT_AU * 1.15   # ~1.75 AU
    ax1.set_xlim(-plot_limit, plot_limit)
    ax1.set_ylim(-plot_limit, plot_limit)

    ax1.set_xlabel("x (AU)", color="white")
    ax1.set_ylabel("y (AU)", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("white")
    ax1.legend(fontsize=8, facecolor="#111111", labelcolor="white")
    ax1.set_title("Experiment 3: Best Satellite Trajectory to Mars", color="white")
    plt.tight_layout()
    plt.show()

    # --- Plot 2: fuel mass vs launch speed ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    speeds    = [r["extra_v"]  for r in results]
    fuels     = [r["fuel_kg"]  for r in results]
    bar_width = (speeds[1] - speeds[0]) * 0.8 if len(speeds) > 1 else 50

    ax2.bar(speeds, fuels, width=bar_width, color="steelblue", alpha=0.8)
    ax2.bar(best["extra_v"], best["fuel_kg"], width=bar_width,
            color="lime", alpha=0.9, label="Best trajectory")

    ax2.set_xlabel("Extra launch speed (m/s)")
    ax2.set_ylabel("Fuel mass (kg)")
    ax2.set_title(f"Experiment 3: Fuel Required vs Launch Speed\n"
                  f"(Payload = {PAYLOAD_MASS:.0f} kg, Isp ≈ 450 s)")
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # --- Animation of the best trajectory ---
    anim_best_trajectory(data_file, best["extra_v"], dt,
                         best["journey_days"], best["min_dist_au"])


def anim_best_trajectory(data_file: str, extra_v: float, dt: int,
                          journey_days: float, min_dist_au: float) -> None:
    """
    Re-runs the best-trajectory simulation and produces a FuncAnimation showing
    the satellite's journey to Mars alongside the live positions of all planets.

    Static elements  : faint orbit rings, Sun, launch marker
    Animated elements: planet dots + name labels, growing satellite trail,
                       satellite dot, closest-approach star (appears on arrival),
                       elapsed-time readout

    Parameters
    ----------
    data_file    : path to planets.json
    extra_v      : extra launch speed (m/s) — the best trajectory value
    dt           : time step in seconds (same value used in run_experiment_3)
    journey_days : days to closest Mars approach (marks the closest-approach frame)
    min_dist_au  : closest approach distance in AU (used in legend label)
    """
    DAY         = 24 * 3600
    MAX_YEARS   = 2
    total_steps = int(MAX_YEARS * EARTH_YEAR / dt)

    # --- Re-run simulation and record all body positions ---
    sim = Simulation(dt=dt, integrator="beeman")
    sim.load_bodies_from_json(data_file)
    sim.initialise_bodies(sun_mass=SUN_MASS)
    satellite = add_satellite(sim, extra_v)

    # Collect metadata before the loop (colours, orbit radii)
    body_colour   = {b.name: b.colour for b in sim.bodies}
    body_orbit_au = {b.name: b.orbital_radius / AU
                     for b in sim.bodies if b.name not in ("Sun", "Satellite")}
    animated_names = [b.name for b in sim.bodies if b.name != "Sun"]

    # Pre-allocate position history arrays for efficiency
    history    = {name: np.empty((total_steps, 2)) for name in animated_names}
    times_days = np.empty(total_steps)

    # Cache body lookups so the inner loop doesn't search each step
    body_map = {b.name: b for b in sim.bodies}

    for step in range(total_steps):
        sim.step()
        for name in animated_names:
            history[name][step] = body_map[name].position / AU
        times_days[step] = sim.time / DAY

    # Subsample: cap at 800 frames so the animation isn't sluggish for small dt
    n_frames  = min(800, total_steps)
    frame_idx = np.linspace(0, total_steps - 1, n_frames, dtype=int)

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    MARS_ORBIT_AU = 1.524
    plot_limit    = MARS_ORBIT_AU * 1.15   # ~1.75 AU — just beyond Mars' orbit
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.set_title(f"Experiment 3: Satellite to Mars  (Δv = {extra_v:.0f} m/s)", color="white")

    # Static: faint orbit rings for each planet
    theta = np.linspace(0, 2 * np.pi, 300)
    for name, r_au in body_orbit_au.items():
        ax.plot(r_au * np.cos(theta), r_au * np.sin(theta),
                color=body_colour[name], linewidth=0.4, alpha=0.3)

    # Static: Sun dot and launch-position marker
    ax.plot(0, 0, "o", color="yellow", markersize=10, zorder=5)
    ax.plot(history["Satellite"][0, 0], history["Satellite"][0, 1],
            "^", color="lime", markersize=7, label="Launch", zorder=5)

    # Animated: satellite trail (grows each frame) and current-position dot
    sat_trail, = ax.plot([], [], color="white", linewidth=0.8, alpha=0.9, zorder=3)
    sat_dot,   = ax.plot([], [], "o", color="cyan", markersize=6, zorder=4)

    # Animated: one dot + floating name label per planet
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

    # Animated: closest-approach star — invisible until the satellite arrives
    closest_marker, = ax.plot([], [], "*", color="red", markersize=12, zorder=6,
                               label=f"Closest approach ({min_dist_au:.3f} AU)")

    # Elapsed-time readout (top-left, fixed to axes coordinates)
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        color="white", fontsize=9, va="top")

    ax.legend(fontsize=8, facecolor="#111111", labelcolor="white", loc="upper right")

    # Frame number at which closest approach occurs
    closest_global_step = int(journey_days * DAY / dt)
    closest_frame       = int(np.argmin(np.abs(frame_idx - closest_global_step)))

    # --- Animation callbacks ---
    def init():
        sat_trail.set_data([], [])
        sat_dot.set_data([], [])
        closest_marker.set_data([], [])
        time_text.set_text("")
        for dot in planet_dots.values():
            dot.set_data([], [])
        return [sat_trail, sat_dot, closest_marker, time_text] + list(planet_dots.values())

    def update(frame_num):
        idx = frame_idx[frame_num]

        # Grow the satellite trail up to the current step
        sat_trail.set_data(history["Satellite"][:idx + 1, 0],
                           history["Satellite"][:idx + 1, 1])
        sat_dot.set_data([history["Satellite"][idx, 0]],
                         [history["Satellite"][idx, 1]])

        # Move each planet dot and its label
        for name, dot in planet_dots.items():
            pos = history[name][idx]
            dot.set_data([pos[0]], [pos[1]])
            planet_labels[name].set_position((pos[0], pos[1] + 0.08))

        # Reveal the closest-approach marker once we reach that frame
        if frame_num >= closest_frame:
            ca = history["Satellite"][frame_idx[closest_frame]]
            closest_marker.set_data([ca[0]], [ca[1]])

        time_text.set_text(f"t = {times_days[idx]:.0f} days")

        return [sat_trail, sat_dot, closest_marker, time_text] + list(planet_dots.values())

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=20, blit=True)
    plt.tight_layout()
    plt.show()
