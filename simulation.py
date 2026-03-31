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
        """Read planet data from a JSON file and populate self.bodies.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            None
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load sun separately for clarity
        sun_data = data["sun"]
        self.bodies.append(Body(
            name=sun_data["name"],
            mass=sun_data["mass"],
            orbital_radius=sun_data["orbital_radius"],
            colour=sun_data["colour"],
        ))

        # Load planets from JSON
        for p in data["planets"]:
            self.bodies.append(Body(
                name=p["name"],
                mass=p["mass"],
                orbital_radius=p["orbital_radius"],
                colour=p["colour"],
            ))

    def add_body(self, body: Body) -> None:
        """Manually add a Body (used for satellites). Caller must set position and velocity."""
        self.bodies.append(body)

    def initialise_bodies(self, sun_mass: float) -> None:
        """
        Set initial positions and velocities for all bodies.

        Sun     : origin, zero velocity
        Planets : positive x-axis at orbital_radius, Keplerian circular speed in +y
                  v_circ = sqrt(G * M_sun / r)

        Center-of-mass velocity is subtracted so the system has zero net momentum.
        prev_acceleration is bootstrapped from a(t=0) for the Beeman method.
        """
        # Load sun
        sun = self.bodies[0]
        sun.position = np.zeros(2)
        sun.velocity = np.zeros(2)
        sun.acceleration = np.zeros(2)
        sun.prev_acceleration = np.zeros(2)

        # Load all other planets and satellites
        for body in self.bodies[1:]:
            r = body.orbital_radius
            v_circ = np.sqrt(self.G * sun_mass / r)
            body.position = np.array([r, 0.0])
            body.velocity = np.array([0.0, v_circ])

        # Zero the center-of-mass velocity so the Sun doesn't drift
        """I initialise all planets moving in the +y direction, and the Sun sitting still at the origin.
        But the system as a whole now has a net momentum since all that +y motion adds up. 
        In a real isolated system, the centre of mass must move at constant velocity forever 
        (Newton's first law applied to the whole system). 
        Since the system initially has a net +y momentum, the entire solar system drifts upward, 
        and from the Sun's perspective it looks like it's being left behind which manifests as the Sun accelerating away.
        
        I was initially very confused by this bug but by checking conservation laws (energy and momentum) the problem was clear,
        and I fixed the bug.

        The fix is to make the total momentum of the system exactly zero, so the centre of mass stays fixed at the origin forever.
        I do that by computing the velocity of the centre of mass:"""

        total_momentum = sum(b.mass * b.velocity for b in self.bodies)
        total_mass = sum(b.mass for b in self.bodies)
        v_com = total_momentum / total_mass

        """
        Now if I subtract the centre of mass velocity from all velocities, the system has zero net momentum. i.e.
        the sun doesnt seem to 'drift' away anymore.
        
        However, this means that the whole solar system will start moving towards -y, but on the scale of the plot
        (which is larger than Neptune's orbital radius), this drift is not noticeable. The physics remains unaffected, and 
        that's what matters
        """
        for body in self.bodies:
            body.velocity -= v_com

        # setting prev_acc = acc for t=0, to prevent running into unnecessary errors.
        # This is only relevant for the Beeman method, which is the only one that uses prev_acc.
        # For t>0, prev_acc is updated in the step() method.
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
            acc_on_j = G * m_i / |r_ij|^2 * r_hat_ij    (r_ij points j -> i)
            acc_on_i = G * m_j / |r_ij|^2 * (-r_hat_ij)

        Newton's third law means forces are equal and opposite,
        but accelerations differ because masses differ.
        """
        #Initialize 0 acceleration for every body
        accelerations = {body.name: np.zeros(2) for body in self.bodies}

        #Calulate acceleration for each pair of bodies (many body sim)
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

        new_accelerations = self.compute_accelerations()

        for body in self.bodies:
            a_next = new_accelerations[body.name]
            body.velocity += (1/6) * (2 * a_next + 5 * body.acceleration - body.prev_acceleration) * self.dt
            body.prev_acceleration = body.acceleration.copy()
            body.acceleration = a_next

    def step_euler_cromer(self) -> None:
        """
        Euler-Cromer update:
            v(t+dt) = v(t) + a(t)*dt
            r(t+dt) = r(t) + v(t+dt)*dt     <- uses NEW velocity

        Accelerations are recomputed AT the new postitions.
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
        stays current.
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
    """
    All of the following methods are required for experiment 2, but were also useful to help debug the simulation,
    as conservation of Energy could be checked.
    """

    def total_kinetic_energy(self) -> float:
        """Sum of (1/2) m v^2 over all bodies."""
        return sum(0.5 * b.mass * float(np.dot(b.velocity, b.velocity)) for b in self.bodies)

    def total_potential_energy(self) -> float:
        """
        U = -G * sum_{i<j} m_i * m_j / |r_ij|
        Unique pairs only to avoid double-counting. Same logic as the acceleration calculation
        """
        pe = 0.0
        for idx_i, body_i in enumerate(self.bodies):
            for body_j in self.bodies[idx_i + 1:]:
                dist = np.linalg.norm(body_i.position - body_j.position)
                pe += -self.G * body_i.mass * body_j.mass / dist
        return pe

    def total_energy(self) -> float:
        """Total energy of the system."""
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

        #Note to self - atan2() returns a value in the range -pi to pi radians, atan() returns from -pi/2 to pi/2

        sun = self.bodies[0]

        for body in self.bodies[1:]:
            if body.is_satellite:
                continue # Ignore satellites

            rel = body.position - sun.position
            angle = np.arctan2(rel[1], rel[0])

            """
            rel is the relative position of the body to the sun, which essentially makes the sun as (0,0) and the position of
            the planet as some (x,y). The angle is then calculated as atan2(y/x)
            
            I think that one problem with this is that the sun is also moving slightly due to gravitational tugs from all 
            other planets. This could lead to slightly inaccurate measurement of period? I couldn't think of a better way to 
            measure periods. This will have to be noted as a limitation.
            """

            # To initialize
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
            """
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            This formula is used to normalise the angle delta to (-pi, pi].
            The edge case, when tracker["prev_angle"] approx = pi, then the next measurement of angle will wrap to 
            something like -pi. This makes delta = roughly 2pi, which is obviously not the case and messes up with the 
            calculation of the cumulative angle ad triggers the period detection before it actually happens.
            This was another bug I ran into.
            
            The normalisation makes sure that delta is always small and positive, and that cumulative angle behaves normally
            """

            tracker["cumulative"] += delta
            tracker["prev_angle"] = angle

            if tracker["cumulative"] >= 2 * np.pi:
                self.periods[body.name] = self.time - tracker["start_time"]

    def print_periods(self, earth_year_seconds: float = 365.25 * 24 * 3600) -> None:
        """Print simulated periods vs NASA reference values."""
        nasa = {
            "Mercury": 0.2408467,
            "Venus":   0.61519726,
            "Earth":   1.0000174,
            "Mars":    1.8808476,
            "Jupiter": 11.862615,
            "Saturn":  29.447498,
            "Uranus":  84.016846,
            "Neptune": 164.79132,
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