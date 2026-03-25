import numpy as np

class Body:
    """
    Represents a single celestial body (planet, sun, or satellite)

    Attributes:
        name : str
        mass : float                        kg
        position : np.ndarray               [x,y] m
        velocity : np.ndarray               [x,y] ms^-1
        acceleration : np.ndarray           current [x,y] ms^-2
        prev_acceleration : np.ndarray      previous [x,y] ms^-2 (required for Beeman Method)
        colour: str                          for matplotlib
        orbital_radius : float                      orbital radius in metres (for initialization)
        is_satellite : bool                 default False
    """

    #Gravitational constant
    G = 6.6743e-11

    def __init__(self, name: str, mass: float, orbital_radius: float, colour: str = 'blue', is_satellite: bool = False):
        self.name = name
        self.mass = mass
        self.orbital_radius = orbital_radius
        self.colour = colour
        self.is_satellite = is_satellite

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.prev_acceleration = np.zeros(2)

    def kinetic_energy(self):
        return 0.5 * self.mass * np.sum(self.velocity**2)

    def __repr__(self) -> str:
        return f"Body({self.name}), mass = {self.mass} kg, Orbital radius = {self.orbital_radius} m"