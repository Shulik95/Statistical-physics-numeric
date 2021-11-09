import math

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

    def time_to_particle(self, other):
        """
        return time until Particle to other Particle.
        :param other: 2nd Particle.
        """
        deltal = other.pos - self.pos
        deltav = other.vel - self.vel
        s = np.dot(deltav, deltal)
        det = s ** 2 - (norm(deltav) ** 2) * (norm(deltal) ** 2 - 4 * (self.rad ** 2))
        if det > 0 > s:
            return -(s + np.sqrt(det)) / (norm(deltav) ** 2)
        else:
            return math.inf

    def time_to_wall(self):
        """
        returns distance
        :return:
        """
        vx, vy = self.vel
        dtwall_x = (1 - self.rad) / vx if vx > 0 else self.rad / abs(vx)
        dtwall_y = (1 - self.rad) / vy if vy > 0 else self.rad / abs(vy)
        return min(dtwall_y, dtwall_x)

    def get_speed(self):
        """
        returns the speed of the current disc.
        """
        return norm(self.vel)


class Simulation:
    """simulation of circular particles in motion"""

    def __init__(self, v_table, p_table, nparticles=4):
        self.v_table = v_table
        self.p_table = p_table
        self.particles = [self.init_particles(p_table, v_table, i) for i in range(nparticles)]
        self.x = np.linspace()

    def init_particles(self, p_table, v_table, row):
        """
        return a new Particle object with position and velocity according
        to tables.
        :param p_table: table of size 4x2 where each row is the x&y position of
        the particle corresponding to that row.
        :param v_table: table of size 4x2 where each row is the vx&vy of
        the particle corresponding to that row.
        """
        pass


if __name__ == '__main__':
    p_table = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    v_table = np.array([[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.34583]])
