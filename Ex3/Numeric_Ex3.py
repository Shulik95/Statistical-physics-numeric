# ------------------------------ imports ------------------------------
import numpy as np
from matplotlib import pyplot
import math
from Ex2.MolDynamics import Particle
import random


# ------------------------------- code --------------------------------

def oneD_random_walk(tend, nparticles=10 ** 6, tstore=None):
    """

    :param tend:
    :param nparticles:
    :param tstore:
    :return:
    """
    pos_dict = {1: [], 5: [], 20: []}
    if tstore is None:
        tstore = [1, 5, 20]
    particle_arr = [Particle(0, 0, 1, 0) for _ in nparticles]  # init particles
    v, flag = 1, nparticles
    while flag > 0:  # not all particles have reached tend
        for particle in particle_arr:
            if particle.t >= tend:
                flag -= 1
            direction = random.choice([1, -1])
            ell = np.random.normal(1.)  # get mean free path
            while ell <= 0:
                ell = np.random.normal(1.)
            delta_x = np.random.exponential(ell)

            for elem in tstore:
                if particle.t < elem < particle.t + abs(delta_x / particle.vel[0]):
                    oneD_helper(particle, elem, pos_dict)
            particle.pos[0] += direction * delta_x
            particle.t += abs(delta_x / particle.vel[0])


def oneD_helper(p, elem, pos_dict):
    dt = elem - p.t
    x_store = p.pos[0] + p.vel[0] * dt
    pos_dict[elem].append(x_store)

