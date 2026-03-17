import numpy as np
import json
from bodies import Body

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
            pass #TODO

        def add_body(self, body: Body) -> None:
            """Add a body to the simulation, for the satellite to Mars experiment"""
            pass #TODO

        def initialze_bodies(self, sun_mass: float) -> None:
            """
            Place each non-sun body on the positive x-axis at its orbital radius.
            Set velocity to Keplerian orbit speed: v = sqrt(G * M_sun / r) in positive y-direction
            :param self:
            :param sun_mass:
            """
            pass #TODO

        # --------------
        # PHYSICS
        # --------------

        def compute_accelerations(self) -> dict[str, np.ndarray]:
            """
            Compute the gravitational acceleration on every body due to all others
            Returns a dictionary of {body_name: acceleration_vector}

            For body j:
            a_j = G * sum_{i!=j} [m_i / |r_ij|^2 * r_hat_ij]
            """
            pass #TODO

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
                pass  # TODO

            def total_potential_energy(self) -> float:
                """
                Gravitational potential energy of the system:
                    U = -G * sum_{i<j} m_i * m_j / |r_ij|
                Note: sum over unique pairs only (i < j) to avoid double-counting.
                """
                pass  # TODO

            def total_energy(self) -> float:
                return self.total_kinetic_energy() + self.total_potential_energy()

            def log_energy(self) -> None:
                """Append (current_time, total_energy) to self.energy_log."""
                self.energy_log.append((self.time, self.total_energy()))

            def write_energy_to_file(self, filepath: str) -> None:
                """Write the energy log to a plain-text file (time, energy per line)."""
                pass  # TODO

            # -----------------------
            # PERIOD DETECTION
            # -----------------------

            def _check_periods(self) -> None:
                """
                Detect when a body completes one full orbit.
                Strategy: track the angle of each body relative to the Sun.
                A full orbit is completed when the cumulative angle crosses 2*pi.
                Records the period in self.periods on first completion.
                """
                pass  # TODO

            def print_periods(self, earth_year_seconds: float = 365.25 * 24 * 3600) -> None:
                """Print orbital periods in Earth years."""
                pass  # TODO