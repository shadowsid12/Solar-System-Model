import numpy as np
import json
from bodies import Body


class Simulation:
    """
    Manages the solar system many-body simulation.

    Responsibilities
    ----------------
    - Load bodies from a JSON file
    - Initialise positions and velocities
    - Step the simulation forward using a chosen integrator
    - Compute accelerations (gravitational many-body)
    - Detect orbital period completion
    - Track and write total system energy to file
    """

    G = 6.6743e-11
    INTEGRATORS = ("beeman", "euler_cromer", "direct_euler")

    def __init__(self, dt: float, integrator: str = "beeman"):
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Invalid integrator: {integrator}")

        self.dt = dt
        self.integrator = integrator
        self.bodies: list[Body] = []
        self.time: float = 0.0

        self.energy_log: list[tuple[float, float]] = []
        self.period_tracker: dict[str, dict] = {}
        self.periods: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_bodies_from_json(self, filepath: str) -> None:
        """Read planet data from a JSON file and populate self.bodies."""
        with open(filepath, "r") as f:
            data = json.load(f)

        sun_data = data["sun"]
        self.bodies.append(Body(
            name=sun_data["name"],
            mass=sun_data["mass"],
            orbital_radius=sun_data["orbital_radius"],
            colour=sun_data["colour"],
        ))

        for p in data["planets"]:
            self.bodies.append(Body(
                name=p["name"],
                mass=p["mass"],
                orbital_radius=p["orbital_radius"],
                colour=p["colour"],
            ))

    def add_body(self, body: Body) -> None:
        """Manually add a Body (used for satellites). Position and velocity must be set by caller."""
        self.bodies.append(body)

    def initialise_bodies(self, sun_mass: float) -> None:
        """
        Set initial positions and velocities for all bodies.

        Sun     : origin, zero velocity
        Planets : positive x-axis at orbital_radius, Keplerian circular speed in +y
                  v_circ = sqrt(G * M_sun / r)

        Centre-of-mass velocity is subtracted so the system has zero net momentum.
        prev_acceleration is bootstrapped from a(t=0) for the Beeman method.
        """
        sun = self.bodies[0]
        sun.position = np.zeros(2)
        sun.velocity = np.zeros(2)
        sun.acceleration = np.zeros(2)
        sun.prev_acceleration = np.zeros(2)

        for body in self.bodies[1:]:
            r = body.orbital_radius
            v_circ = np.sqrt(self.G * sun_mass / r)
            body.position = np.array([r, 0.0])
            body.velocity = np.array([0.0, v_circ])

        # Zero the centre-of-mass velocity so the Sun doesn't drift
        total_momentum = sum(b.mass * b.velocity for b in self.bodies)
        total_mass = sum(b.mass for b in self.bodies)
        v_com = total_momentum / total_mass
        for body in self.bodies:
            body.velocity -= v_com

        # Bootstrap accelerations at t=0
        accelerations = self.compute_accelerations()
        for body in self.bodies:
            body.acceleration = accelerations[body.name]
            body.prev_acceleration = accelerations[body.name].copy()

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def compute_accelerations(self) -> dict[str, np.ndarray]:
        """
        Gravitational acceleration on every body due to all others.

        For each unique pair (i, j):
            acc_on_j = G * m_i / |r_ij|^2 * r_hat_ij   (r_ij points j -> i)
            acc_on_i = G * m_j / |r_ij|^2 * (-r_hat_ij)

        Newton's third law means forces are equal and opposite,
        but accelerations differ because masses differ.
        """
        accelerations = {body.name: np.zeros(2) for body in self.bodies}

        for idx_i, body_i in enumerate(self.bodies):
            for body_j in self.bodies[idx_i + 1:]:
                r_ij = body_i.position - body_j.position
                dist = np.linalg.norm(r_ij)
                r_hat = r_ij / dist

                acc_on_j = self.G * body_i.mass / dist**2 * r_hat
                acc_on_i = self.G * body_j.mass / dist**2 * (-r_hat)

                accelerations[body_j.name] += acc_on_j
                accelerations[body_i.name] += acc_on_i

        return accelerations

    # ------------------------------------------------------------------
    # Integrators
    # ------------------------------------------------------------------

    def step_beeman(self) -> None:
        """
        Beeman update:
            r(t+dt) = r(t) + v(t)*dt + (1/6)[4a(t) - a(t-dt)] * dt^2
            a(t+dt) = compute_accelerations() at new positions
            v(t+dt) = v(t) + (1/6)[2a(t+dt) + 5a(t) - a(t-dt)] * dt
        """
        for body in self.bodies:
            body.position += (body.velocity * self.dt
                              + (1/6) * (4 * body.acceleration - body.prev_acceleration) * self.dt**2)

        new_accels = self.compute_accelerations()

        for body in self.bodies:
            a_next = new_accels[body.name]
            body.velocity += (1/6) * (2 * a_next + 5 * body.acceleration - body.prev_acceleration) * self.dt
            body.prev_acceleration = body.acceleration.copy()
            body.acceleration = a_next

    def step_euler_cromer(self) -> None:
        """
        Euler-Cromer update:
            v(t+dt) = v(t) + a(t)*dt
            r(t+dt) = r(t) + v(t+dt)*dt     <- uses NEW velocity

        Acceleration is recomputed at the new positions so body.acceleration
        stays current for energy calculations and period detection.
        """
        for body in self.bodies:
            body.velocity += body.acceleration * self.dt
            body.position += body.velocity * self.dt

        new_accels = self.compute_accelerations()
        for body in self.bodies:
            body.acceleration = new_accels[body.name]

    def step_direct_euler(self) -> None:
        """
        Direct (Forward) Euler update:
            r(t+dt) = r(t) + v(t)*dt         <- uses OLD velocity
            v(t+dt) = v(t) + a(t)*dt

        Acceleration is recomputed at the new positions so body.acceleration
        stays current. Energy will drift upward — this is expected and is the
        point of Experiment 2.
        """
        for body in self.bodies:
            body.position += body.velocity * self.dt
            body.velocity += body.acceleration * self.dt

        new_accels = self.compute_accelerations()
        for body in self.bodies:
            body.acceleration = new_accels[body.name]

    def step(self) -> None:
        """Advance simulation by one time step using self.integrator."""
        if self.integrator == "beeman":
            self.step_beeman()
        elif self.integrator == "euler_cromer":
            self.step_euler_cromer()
        elif self.integrator == "direct_euler":
            self.step_direct_euler()

        self.time += self.dt
        self.check_periods()

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def total_kinetic_energy(self) -> float:
        """Sum of (1/2) m v^2 over all bodies."""
        return sum(0.5 * b.mass * float(np.dot(b.velocity, b.velocity)) for b in self.bodies)

    def total_potential_energy(self) -> float:
        """
        U = -G * sum_{i<j} m_i * m_j / |r_ij|
        Unique pairs only to avoid double-counting.
        """
        pe = 0.0
        for idx_i, body_i in enumerate(self.bodies):
            for body_j in self.bodies[idx_i + 1:]:
                dist = np.linalg.norm(body_i.position - body_j.position)
                pe += -self.G * body_i.mass * body_j.mass / dist
        return pe

    def total_energy(self) -> float:
        return self.total_kinetic_energy() + self.total_potential_energy()

    def log_energy(self) -> None:
        """Append (current_time, total_energy) to self.energy_log."""
        self.energy_log.append((self.time, self.total_energy()))

    def write_energy_to_file(self, filepath: str) -> None:
        """Write the energy log to a CSV file (time in seconds, energy in joules)."""
        with open(filepath, "w") as f:
            f.write("time_s,total_energy_J\n")
            for t, e in self.energy_log:
                f.write(f"{t:.6e},{e:.6e}\n")

    # ------------------------------------------------------------------
    # Period detection
    # ------------------------------------------------------------------

    def check_periods(self) -> None:
        """
        Detect when a body completes one full orbit using cumulative angle tracking.

        Each non-Sun, non-satellite body gets an entry in self.period_tracker:
            {
                "prev_angle": float,    # atan2 angle at previous time step
                "cumulative": float,    # total angle accumulated (radians)
                "start_time": float,    # self.time when tracking began
            }

        delta is normalised into (-pi, pi] to handle the atan2 wrap-around.
        Period is recorded when cumulative angle first reaches 2*pi.
        """
        sun = self.bodies[0]

        for body in self.bodies[1:]:
            if body.is_satellite:
                continue

            rel = body.position - sun.position
            angle = np.arctan2(rel[1], rel[0])

            if body.name not in self.period_tracker:
                self.period_tracker[body.name] = {
                    "prev_angle": angle,
                    "cumulative": 0.0,
                    "start_time": self.time,
                }
                continue

            if body.name in self.periods:
                continue

            tracker = self.period_tracker[body.name]
            delta = angle - tracker["prev_angle"]
            delta = (delta + np.pi) % (2 * np.pi) - np.pi

            tracker["cumulative"] += delta
            tracker["prev_angle"] = angle

            if tracker["cumulative"] >= 2 * np.pi:
                self.periods[body.name] = self.time - tracker["start_time"]

    def print_periods(self, earth_year_seconds: float = 365.25 * 24 * 3600) -> None:
        """Print simulated periods vs NASA reference values."""
        nasa = {
            "Mercury": 0.2409,
            "Venus":   0.6152,
            "Earth":   1.0000,
            "Mars":    1.8809,
            "Jupiter": 11.862,
            "Saturn":  29.457,
            "Uranus":  84.011,
            "Neptune": 164.79,
        }

        print(f"\n{'Body':<10} {'Simulated (yr)':>15} {'NASA (yr)':>12} {'Error (%)':>10}")
        print("-" * 50)

        for body in self.bodies[1:]:
            if body.is_satellite or body.name not in self.periods:
                continue
            simulated_yr = self.periods[body.name] / earth_year_seconds
            reference    = nasa.get(body.name, float("nan"))
            error        = abs(simulated_yr - reference) / reference * 100
            print(f"{body.name:<10} {simulated_yr:>15.4f} {reference:>12.4f} {error:>9.2f}%")