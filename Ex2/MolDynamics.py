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

    def __init__(self, v_table, p_table, nparticles=4, tot_v=2, dtstore=1, N=10 ** 7):
        self.v_table = v_table
        self.p_table = p_table
        self.particles = [self.init_particles(i) for i in range(nparticles)]
        self.x = np.linspace()
        self.nparticles = nparticles
        self.tot_v = tot_v
        self.t = 0
        self.collision = 0
        self.dtstore = dtstore
        self.N = N

    def init_particles(self, row):
        """
        return a new Particle object with position and velocity according
        to corresponding tables.
        """
        x, y = self.p_table[row]
        vx, vy = self.v_table[row]
        return Particle(x, y, vx, vy)

    def advance(self):
        """
        advances the system the following way:
        for each time t:
            (1) for each particles check time to collision with wall
            (2) check time for collision between all particles
            (3) define dt = min{dtwall, dtcoll}
            (4) advance position of all particles according to dt
            (5) update time by dt and update speed of particle(s)
        """
        # find minimal time until next collision with wall or particle
        dtwall, p0 = min([(self.particles[i].time_to_wall(), i) for i in range(self.nparticles)], key=lambda t: t[0])

        min_dtcoll, p1, p2 = self.find_min_dtcoll()
        dt = min(dtwall, min_dtcoll)

        # update particle location
        for particle in self.particles:
            particle.pos[0] = particle.pos[0] + particle.vel[0] * dt
            particle.pos[1] = particle.pos[1] + particle.vel[1] * dt

        # update time and velocity of colliding particles
        self.t = self.t + dt
        if dtwall < min_dtcoll:
            self.update_vel_wall(p0)
        else:
            self.update_vel_coll(p1, p2)

    def find_min_dtcoll(self):
        """
        finds the minimal time until collision between two particles.
        :return: minimal collision time, idx of the two colliding particles.
        """
        min_dtcoll, p1, p2 = math.inf, None, None
        for i in range(self.nparticles):
            for j in range(i + 1, self.nparticles):
                dtcoll = self.particles[i].time_to_particle(self.particles[j])
                if dtcoll < min_dtcoll:
                    min_dtcoll = dtcoll
                    p1, p2 = i, j
        return min_dtcoll, p1, p2

    def update_vel_wall(self, p0):
        """
        updates particle velocity after collision with wall
        :param p0: idx of the particle which collided
        """
        if self.particles[p0].pos[0] in [0, 1]:
            self.particles[p0].vel[0] = -self.particles[p0].vel[0]
        else:
            self.particles[p0].vel[1] = -self.particles[p0].vel[1]

    def update_vel_coll(self, p1, p2):
        """
        updates velocity for particles after collision.
        :param p1: idx of 1st particle
        :param p2: idx of 2nd particle
        """
        delta_pos = self.particles[p1].pos - self.particles[p2].pos
        delta_vel = self.particles[p1].vel - self.particles[p2].vel
        deltal = norm(delta_pos)
        # deltav = norm(delta_vel)
        ex = delta_pos[0] / deltal
        ey = delta_pos[1] / deltal
        s_v = delta_vel[0] * ex + delta_vel[1] * ey
        self.particles[p1].vel[0] = self.particles[p1].vel[0] + ex * s_v
        self.particles[p1].vel[1] = self.particles[p1].vel[1] + ey * s_v
        self.particles[p2].vel[0] = self.particles[p2].vel[0] - ex * s_v
        self.particles[p2].vel[1] = self.particles[p2].vel[1] - ey * s_v


if __name__ == '__main__':
    p_table = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    v_table = np.array([[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.34583]])
