import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


class Particle:
    """A circular particle of unit mass with position and velocity"""

    def __init__(self, x, y, vx, vy, rad=0.15):
        self.pos = np.array([x, y])
        self.vel = np.array([vx, vy])
        self.rad = rad

    def advance(self, dt):
        """
        advance the particles position according to its velocity, boundary
        conditions are newtonian, not periodic.
        :param dt: the time which
        """
        self.pos = self.pos + self.vel * dt

    def distance_to_particle(self, other):
        """
        return distance from this Particle to other Particle.
        :param other: 2nd Particle.
        """
        pass

    def get_speed(self):
        return norm(self.vel)
