import numpy as np
import json
from bodies import Body


class Simulation:
    """
    Manages a gravitational N-body simulation of the solar system.

    This class handles the initialisation, physical integration, and data logging
    for a system of celestial bodies. It supports multiple integration methods
    including Beeman, Euler-Cromer, and Direct Euler, and provides utilities
    for energy conservation tracking and orbital period detection.

    Attributes:
        G (float): Gravitational constant (6.6743e-11 m^3 kg^-1 s^-2).
        INTEGRATORS (tuple): Supported numerical integration method names.
        dt (float): Simulation time step in seconds.
        integrator (str): The active integration algorithm.
        bodies (list[Body]): List of Body objects in the simulation.
        time (float): Elapsed simulation time in seconds.
        energy_log (list): Historical record of (time, total_energy) tuples.
        period_tracker (dict): Internal state for tracking cumulative orbital angles.
        periods (dict): Recorded orbital periods for each body in seconds.
    """

    G = 6.6743e-11
    INTEGRATORS = ("beeman", "euler_cromer", "direct_euler")

    def __init__(self, dt: float, integrator: str = "beeman"):
        """
        Initializes the Simulation instance.

        Args:
            dt (float): Time step for each integration increment.
            integrator (str): Numerical method to use. Must be one of
                        ('beeman', 'euler_cromer', 'direct_euler'). Defaults to "beeman".

        Raises:
            ValueError: If the provided integrator is not in self.INTEGRATORS.
        """
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
        """
        Parses a JSON configuration file to populate the simulation bodies.

        Args:
            filepath (str): Path to the JSON file containing 'sun' and 'planets' keys.
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
        """
        Appends a Body object to the simulation manually.

        Args:
            body (Body): The body instance to add. Position and velocity
                should be pre-configured if not using initialise_bodies.
        """
        self.bodies.append(body)

    def initialise_bodies(self, sun_mass: float) -> None:
        """
        Sets initial kinematic states for all bodies and adjusts for zero net momentum.

        Initializes the Sun at the origin and planets at their respective orbital
        radii with Keplerian circular velocities. To prevent system drift, the
        Center of Mass (CoM) velocity is calculated and subtracted from all bodies,
        ensuring the total momentum of the system is zero.

        Args:
            sun_mass (float): Mass of the central star in kilograms.
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

        # Zero the center-of-mass velocity so the Sun doesn't drift.
        '''
        I initialise all planets moving in the +y direction, and the Sun sitting still at the origin.
        But the system as a whole now has a net momentum since all that +y motion adds up.
        In a real isolated system, the centre of mass must move at constant velocity forever
        (Newton's first law applied to the whole system).
        Since the system initially has a net +y momentum, the entire solar system drifts upward,
        and from the Sun's perspective it looks like it's being left behind which manifests as the Sun accelerating away.
        
        I was initially very confused by this bug but by checking conservation laws (energy and momentum) the problem was clear,
        and I fixed the bug.

        The fix is to make the total momentum of the system exactly zero, so the centre of mass stays fixed at the origin forever.
        I do that by computing the velocity of the centre of mass:
        '''

        total_momentum = sum(b.mass * b.velocity for b in self.bodies)
        total_mass = sum(b.mass for b in self.bodies)
        v_com = total_momentum / total_mass

        '''
        Now if I subtract the center of mass velocity from all velocities, the system has zero net momentum. i.e.
        the sun doesn't seem to 'drift' away anymore.
        
        However, this means that the whole solar system will start moving towards -y, but on the scale of the plot
        (which is larger than Neptune's orbital radius), this drift is not noticeable. The physics remains unaffected,
        and that's what matters. Relative to the sun, the velocities of each planet are unchanged.
        '''

        for body in self.bodies:
            body.velocity -= v_com

        '''
        Setting prev_acc = acc for t=0, to prevent running into unnecessary errors.
        This is only relevant for the Beeman method, which is the only one that uses prev_acc.
        For t>0, prev_acc is updated in the step() method.
        '''

        accelerations = self.compute_accelerations()
        for body in self.bodies:
            body.acceleration = accelerations[body.name]
            body.prev_acceleration = accelerations[body.name].copy()

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def compute_accelerations(self) -> dict[str, np.ndarray]:
        """
        Calculates the instantaneous gravitational acceleration for all bodies.

        Uses a vectorised approach to compute pairwise interactions because for longer simulations, looping over each pair
        is extremely slow, compared to this method. For N bodies,
        displacements are calculated in an (N, N, 2) matrix. Diagonal terms
        (self-interaction) are masked to prevent division by zero.

        Returns:
            dict[str, np.ndarray]: Mapping of body names to their 2D acceleration vectors (m/s^2).
        """

        """
        For each unique pair (i, j):
            acc_on_j = G * m_i / |r_ij|^2 * r_hat_ij
            acc_on_i = G * m_j / |r_ij|^2 * (-r_hat_ij)

        Newton's third law: forces equal and opposite, accelerations differ
        because masses differ.

        Implementation
        --------------
        positions: (N, 2) array of all body positions for N bodies
                   each row is for 1 body, and the 2 columns are x and y
                   => positions[i][1] = y coordinate of body i

        distances: (N, N, 2) array of Euclidean distances between each pair
                    here, (i,j,1) is the x-component of the distance vector between bodies i and j
                    the order of i and j doesn't matter as for displacement, the distances will be squared for x and y distance.
                    To ensure the force is attractive, r_ji is the vector pointing from j to i, so the force is negative.
                    the diagonal entries are set to 1 to avoid division-by-zero (self-interaction terms are zeroed out separately)

        masses: (N,) array of all body masses [1d array, so a vector]

        diff[i, j] = positions[i] - positions[j], shape (N, N, 2)
            The displacement vector FROM body j TO body i.

        dist[i, j] = |diff[i, j]|, shape (N, N)
            Euclidean distance between each pair. Diagonal is set to 1
            to avoid division-by-zero (self-interaction terms are zeroed
            out separately via the mass_matrix).

        The acceleration on body i due to all others:
            a[i] = G * sum_j [ m_j / dist[i,j]^2 * r_hat[i,j] ]
                 = G * sum_j [ m_j / dist[i,j]^3 * diff[i,j] ]

        mass_matrix[i, j] = m_j with diagonal = 0 so self-interaction
        contributes nothing to the sum.

        acc_matrix[i, j] = G * m_j / dist[i,j]^3 * diff[i,j]   shape (N, N, 2)
        acc[i] = sum_j acc_matrix[i, j]                          shape (N, 2)
        """
        N         = len(self.bodies)
        positions = np.array([b.position for b in self.bodies])   # (N, 2)
        masses    = np.array([b.mass     for b in self.bodies])   # (N,)

        # diff[i, j] = position[i] - position[j]  — vector from j to i
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
        """
        np.newaxis creates a new axis in the positions array, so that diff[i, j] is a 3D array of shape (N, N, 2)
        (essential its (N,1,2) - (1,N,2) = (N,N,2). np will stretch the newaxis to match length.)
        this means that diff[i,j] is a vector pointing from body j to body i. Note that newaxis creates a pseudo-index,
        no values are stored.
        """


        # Euclidean distance for each pair
        dist = np.linalg.norm(diff, axis=2)   # (N, N)

        # Set diagonal to 1 to avoid division-by-zero on self-interaction terms.
        # These entries are multiplied by zero via the mass_matrix so they
        # never contribute to the final accelerations.
        np.fill_diagonal(dist, 1.0)

        # mass_matrix[i, j] = mass of body j, zero on diagonal (no self-force)
        mass_matrix = np.where(np.eye(N, dtype=bool), 0.0, masses[np.newaxis, :])
        """
        np.eye makes a identity matrix where bool means diagonal is True.
        np.where is used so that we can then set those diagonals as 0.0 (to prevent self-interaction terms)
        """

        # acc_matrix[i, j] = G * m_j / |r_ij|^3 * diff[i,j]
        # Dividing by dist^3 (not dist^2) because diff already carries one
        # factor of dist that would otherwise be in r_hat = diff / dist.
        coeff = self.G * mass_matrix / dist**3   # (N, N)
        """
        Essentially calculates the force on body j due to body i
        This will be multiplied by the distance vector diff[i,j] to get the acceleration on body i.
        """

        # Sum over j to get net acceleration on each body i.
        # Gravity is attractive: acceleration on i points FROM i TOWARD j,
        # which is the direction of (pos[j] - pos[i]) = -diff[i,j].
        # Hence the negative sign.
        acc_array = -np.einsum("ij,ijk->ik", coeff, diff)   # (N, 2)

        return {body.name: acc_array[i] for i, body in enumerate(self.bodies)}

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

        Accelerations are recomputed AT the new positions.
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
    # All of the following methods are required for Experiment 2, but were also
    # useful to help debug the simulation, as conservation of energy could be checked.

    def total_kinetic_energy(self) -> float:
        """
        Calculates the total kinetic energy of all bodies in the system:
        KE = 1/2 * m * v^2

        Returns:
            float: Total kinetic energy in Joules.
        """
        return sum(0.5 * b.mass * float(np.dot(b.velocity, b.velocity)) for b in self.bodies)

    def total_potential_energy(self) -> float:
        """
        Calculates the total gravitational potential energy of the system.
        Computes the sum of -G*m1*m2/r for all unique pairs to avoid double-counting.
        Returns:
            float: Total potential energy in Joules.
        """
        pe = 0.0
        for idx_i, body_i in enumerate(self.bodies):
            for body_j in self.bodies[idx_i + 1:]:
                dist = np.linalg.norm(body_i.position - body_j.position)
                pe += -self.G * body_i.mass * body_j.mass / dist
        return pe

    def total_energy(self) -> float:
        """
        Exports the energy log to a CSV file.

        Args:
            filepath (str): Destination path for the CSV output.
        """
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
        Monitors cumulative angular displacement to detect completed orbits.

        Uses atan2 to track the angle of each body relative to the central Sun.
        Changes in angle are normalised to the range (-pi, pi] to handle
        quadrant wrapping. A period is recorded when the cumulative angle
        reaches 2*pi.
        """
        # Note: atan2() returns a value in the range -pi to pi radians, whereas atan() returns from -pi/2 to pi/2.

        sun = self.bodies[0]

        for body in self.bodies[1:]:
            if body.is_satellite:
                continue  # Ignore satellites

            rel = body.position - sun.position
            angle = np.arctan2(rel[1], rel[0])

            # rel is the relative position of the body to the sun, which essentially makes the sun as (0,0)
            # and the position of the planet as some (x,y). The angle is then calculated as atan2(y/x).
            #
            # One limitation: the Sun is also moving slightly due to gravitational tugs from all other planets.
            # This could lead to slightly inaccurate measurement of period. This will be noted as a limitation
            # in the report.

            # Initialise tracker on first call
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

            # The normalisation formula above maps delta to (-pi, pi].
            # The edge case: when tracker["prev_angle"] is approximately pi, the next measurement
            # of angle will wrap to something like -pi. Without normalisation, delta would be roughly
            # -2pi — which makes the cumulative angle decrease suddenly and trigger period detection
            # at the wrong time. This was a bug I ran into.
            # The normalisation ensures delta is always small and the cumulative angle grows smoothly.

            tracker["cumulative"] += delta
            tracker["prev_angle"] = angle

            if tracker["cumulative"] >= 2 * np.pi:
                self.periods[body.name] = self.time - tracker["start_time"]

    def print_periods(self, earth_year_seconds: float = 365.25 * 24 * 3600) -> None:
        """
        Displays a comparison between simulated periods and NASA reference values.

        Args:
            earth_year_seconds (float): Conversion factor for seconds to Earth years.
        """
        # More precise NASA reference periods (sidereal, in Earth years)
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