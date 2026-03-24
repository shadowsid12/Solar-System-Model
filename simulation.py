import numpy as np
import json
from pathlib import Path
from bodies import Body

DATA_FILE = Path(__file__).parent / "data" / "planets.json"

class Simulation:
    """
    Manages the solar system many-body simulation

    Responsibilities
    ----------------
    - Load bodies from JSON file
    - Initialize positions and velocities
    - Step the simulation forward using a chosen integration method
    - Compute accelerations
    - Detect orbital period completion
    - Track the total energy of the solar system
    """

    #Gravitational constant
    G = 6.6743e-11

    #Available integrators
    INTEGRATORS = ("beeman", "euler_cromer", "direct_euler")

    def __init__(self, dt: float, integrator: str = "euler_cromer"):
        """
        Parameters
        ----------
        :param dt: time step in seconds
        :param integrator: one of INTEGRATORS
        """
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Invalid integrator: {integrator}")

        self.dt = dt
        self.integrator = integrator
        self.bodies: list[Body] = []
        self.time: float = 0.0

        #Energy log: list of (time, energy)
        self.energy_log: list[tuple[float, float]] = []

        #Period tracking: dict of {body: period or None}
        self.period_log: dict[Body, float | None] = {}

        #period tracker: stores the angle of each planet and tracks the period}
        self.angle_tracker: dict = {}



        def load_bodies_from_json(self, filepath: str) -> None:
            """Read planet data from a JSON file and populate self.bodies"""
            with open(DATA_FILE, "r") as f:
                data = json.load(f)

            # Sun first — orbital_radius is 0 but stored for consistency
            sun_data = data["sun"]
            self.bodies.append(Body(
                name=sun_data["name"],
                mass=sun_data["mass"],
                orbital_radius=sun_data["orbital_radius"],
                colour=sun_data["colour"]
            ))

            # Planets in file order
            for p in data["planets"]:
                self.bodies.append(Body(
                    name=p["name"],
                    mass=p["mass"],
                    orbital_radius=p["orbital_radius"],
                    colour=p["colour"]
                ))

        def add_body(self, body: Body) -> None:
            """
            Manually add a Body to the simulation.
            Used for satellites in experiments — call after initialise_bodies().
            The body's position and velocity must already be set by the caller.
            """
            pass #TODO

        def initialize_bodies(self, sun_mass: float) -> None:
            """
            Set initial positions and velocities for all bodies.

        Sun:
            position = origin, velocity = zero, accelerations = zero

        Each planet:
            position = (orbital_radius, 0)   — positive x-axis
            velocity = (0, v_circ)           — positive y-direction, where v_circ = sqrt(G * M_sun / r)  [Keplerian circular orbit]
            prev_acceleration = acceleration = compute_accelerations() result (Beeman requires a(t-dt) at t=0; we bootstrap it from a(t=0))
            """

            #Initialize the sun
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

            #Now compute acceleration due to grav force for all bodies now that they have been positioned
            # Also set prev_acceleration to self.compute_accelerations() result at t=0
            accelerations = self.compute_accelerations()

            for body in self.bodies:
                body.acceleration = accelerations[body.name]
                body.prev_acceleration = accelerations[body.name].copy()

        # --------------
        # PHYSICS
        # --------------

        def compute_accelerations(self) -> dict[str, np.ndarray]:
            """
            Compute the gravitational acceleration on every body due to all others.
            Returns a dict {body_name: acceleration_vector}.

            For body j:
                a_j = G * sum_{i != j} [ m_i / |r_ij|^2 * r_hat_ij ]
            where r_ij = r_i(t) - r_j(t)  (vector from j to i)
            """

            #Initialize all accelerations to 0
            accelerations = {body.name: np.zeros(2) for body in self.bodies}

            # Iterate over every unique pair (i, j) with i < j
            # Newton's third law: force on j from i = -force on i from j

            for index_i, body_i in enumerate(self.bodies):
                for body_j in self.bodies[index_i+1:]:
                    r_ij = body_i.position - body_j.position    # Vector from j to i
                    dist = np.linalg.norm(r_ij)                 # Distance between j and i
                    r_hat = r_ij / dist                         # Unit vector from j to i

                    # Magnitude of acceleration: G * m / |r|^2
                    acc_on_j = self.G * body_i.mass / dist**2 * r_hat
                    acc_on_i = -acc_on_j # Newtons 3rd law

                    accelerations[body_j.name] += acc_on_j
                    accelerations[body_i.name] += acc_on_i

            return accelerations

        def step_beeman(self) -> None:
            """
            Beeman Update:
                r(t+dt) = r(t) + v(t) * dt + (1/6)[4a(t) - a(t-dt)] * dt^2
                v(t+dt) = v(t) + (1/6)[2a(t+dt) + 5a(t) - a(t-dt)] * dt
                a(t+dt) = compute_accelerations() at new positions
            """
            pass #TODO

        def step_euler_cromer(self) -> None:
            """
            Euler-Cromer update:
                v(t+dt) = v(t) + a(t)*dt
                r(t+dt) = r(t) + v(t+dt)*dt     <- uses NEW velocity
            """
            pass  # TODO

        def step_direct_euler(self) -> None:
            """
            Direct (Forward) Euler update:
                r(t+dt) = r(t) + v(t)*dt         <- uses OLD velocity
                v(t+dt) = v(t) + a(t)*dt
            Optionally: swap a(t) -> a(t+dt) in velocity update as an experiment.
            """
            pass  # TODO

        def step(self) -> None:
            """Advance simulation by one time step using self.integrator."""
            if self.integrator == "beeman":
                self.step_beeman()
            elif self.integrator == "euler_cromer":
                self.step_euler_cromer()
            elif self.integrator == "direct_euler":
                self.step_direct_euler()

            self.time += self.dt
            self._check_periods()

        # --------------
        # ENERGY
        # --------------

        def total_kinetic_energy(self) -> float:
            """Sum of (1/2) m v^2 over all bodies."""
            return sum(0.5 * b.mass * float(np.dot(b.velocity, b.velocity)) for b in self.bodies)

        def total_potential_energy(self) -> float:
            """
            Gravitational potential energy of the system:
                U = -G * sum_{i<j} m_i * m_j / |r_ij|
            Note: sum over unique pairs only (i < j) to avoid double-counting.
            """
            pe = 0.0
            for index_i, body_i in enumerate(self.bodies):
                for body_j in self.bodies[index_i + 1:]:
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

        # -----------------------
        # PERIOD DETECTION
        # -----------------------

        def check_periods(self) -> None:
            """
            Detect when a body completes one full orbit using cumulative angle tracking.

            Each non-Sun, non-satellite body gets an entry in self.period_log:
                {
                    "prev_angle": float,       # atan2 angle at the previous time step
                    "cumulative": float,       # total angle accumulated so far (radians)
                    "start_time": float,       # self.time when tracking began
                }

            Every call, the angular step (delta_angle) since the last step is computed. Because atan2
            wraps at ±pi, delta_angle is normalised into (-pi, pi] to handle the wrap-around correctly

            When the cumulative angle first crosses 2*pi, the period is recorded and the tracker entry is
            removed so it does not trigger again.
            """
            sun = self.bodies[0]

            for body in self.bodies[1:]:
                # Satellites are not tracked for orbital periods
                if body.is_satellite:
                    continue

                # Position relative to the Sun
                rel = body.position - sun.position
                angle = np.arctan2(rel[1], rel[0])  # in (-pi, pi]

                if body.name not in self.period_log:
                    # For the first call, to initialize the tracker, nothing to compare yet
                    self.period_log[body.name] = {
                        "prev_angle": angle,
                        "cumulative": 0.0,
                        "start_time": self.time,
                    }
                    continue

                # Already completed — skip
                if body.name in self.periods:
                    continue

                tracker = self.period_log[body.name]

                # Angular step since the last time-step, normalized to (-pi, pi]
                # Without normalization, the atan2 wrap from ~pi to ~-pi would appear as a huge jump instead
                # of a tiny positive step
                delta = angle - tracker["prev_angle"]
                delta = (delta + np.pi) % (2 * np.pi) - np.pi

                tracker["cumulative"] += delta
                tracker["prev_angle"] = angle

                # Full orbit is completed when the cumulative angle first reaches 2*pi
                if tracker["cumulative"] >= 2 * np.pi:
                    self.periods[body.name] = self.time - tracker["start_time"]


        def print_periods(self, earth_year_seconds: float = 365.25 * 24 * 3600) -> None:
            """
                    Print simulated orbital periods alongside NASA reference values.
                    NASA reference periods (in Earth years):
                        Mercury: 0.2409, Venus: 0.6152, Earth: 1.0000,
                        Mars: 1.8809, Jupiter: 11.862
                    """
            nasa = {
                "Mercury": 0.2409,
                "Venus": 0.6152,
                "Earth": 1.0000,
                "Mars": 1.8809,
                "Jupiter": 11.862,
            }

            print(f"\n{'Body':<10} {'Simulated (yr)':>15} {'NASA (yr)':>12} {'Error (%)':>10}")
            print("-" * 50)

            for body in self.bodies[1:]:
                if body.is_satellite or body.name not in self.periods:
                    continue
                simulated_yr = self.periods[body.name] / earth_year_seconds
                reference = nasa.get(body.name, float("nan"))
                error = abs(simulated_yr - reference) / reference * 100
                print(f"{body.name:<10} {simulated_yr:>15.4f} {reference:>12.4f} {error:>9.2f}%")