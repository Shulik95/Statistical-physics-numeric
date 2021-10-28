import numpy as np
import sklearn
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import normalize


def numeric_warmup(N_a=100, N_b=100, q_a=300, q_b=0):
    """
    simulates the energy distribution of two Einstein solids.
    :param N_a: integer - number of particles in first solid.
    :param N_b: integer - number of particles in the 2nd solid.
    :param q_a: integer - number of energy "units" in 1st solid.
    :param q_b: integer - number of energy "units" in 2nd solid.
    """
    q_arr = [0] * (N_a + N_b)
    q_a_arr, q_b_arr = [], []
    counter = 0
    # creating initial state
    for i in range(q_a):
        rand_idx = np.random.randint(0, N_a)
        q_arr[rand_idx] += 1

    # simulate dynamic
    for k in range(10 ** 5):
        i = np.random.randint(0, N_a + N_b)
        j = np.random.randint(0, N_a + N_b)
        if q_arr[i] != 0:  # energy exists
            q_arr[i] -= 1
            q_arr[j] += 1
        q_a_arr.append(sum(q_arr[:N_a]))
        q_b_arr.append(sum(q_arr[N_a:]))

    # plot data
    plt.plot(range(10 ** 5), q_a_arr, 'r', label="Total energy in solid A")
    plt.plot(range(10 ** 5), q_b_arr, 'b', label="Total energy in solid B")
    plt.title(r'$q_a & q_b$ vs. # of iterations')
    plt.legend(), plt.grid()


def metropolis_algo(N=100, theta=2.5):
    """
    Implements the metropolis algorithm which is based on Boltzmann's distribution
    :param N: integer - number of particles in Einstein's solid
    :param theta: float - represents the temperature of the heat bath
    """
    solid = [0] * N
    func = np.exp(-1 / theta)
    q_tot, q_single_part = [], []
    bins = np.array([int(x) for x in range(int(10 * theta) + 1)])

    # simulate dynamics
    for j in range(10 ** 7):
        i = np.random.randint(0, N)  # choose particle
        del_q = random.choice([-1, 1])
        update_solid(del_q, func, i, solid)
        q_tot.append(sum(solid))  # track total energy
        q_single_part.append(solid[0])  # track energy of first particle
        if j in [2 * (10 ** 6), 4 * (10 ** 6), 6 * (10 ** 6), 8 * (10 ** 6)]:
            plt.hist(solid, bins=bins, lw=1, histtype='step',
                     label=str(j) + " Iterations")
            # plt.title("Histogram after " + str(j) + " iterations")
            # plt.savefig("histogram after " + str(j) + " iteration.jpeg")
            # plt.show()
    plt.hist(solid, bins=bins, linewidth=1, histtype='step', label=str(10 ** 7) + " Iterations")
    plt.savefig("histograms_united.jpeg"), plt.legend(loc='upper right'), plt.xlabel("Units of energy"), plt.ylabel("# of particles")
    plt.show()

    # plt.title("Final histogram"), plt.savefig('final_histogram.jpeg')

    plt.show()
    plt.hist(q_single_part, bins=bins, color='skyblue', density=True, align='mid', edgecolor='black', linewidth=1)
    plt.title("Single particle"), plt.xlabel("q"), plt.ylabel("% of steps"), plt.savefig(
        "Single particle cumulative histogram")
    plt.show()
    plt.plot(range(10 ** 7),np.array(q_tot), color='lightcoral'), plt.xlabel("# of steps"), plt.ylabel(r'$q_t$'), plt.title(r'$q_t$ vs. # of steps')
    plt.savefig("q_t vs. steps.jpeg"), plt.show()


def update_solid(del_q, func, i, solid):
    """
    updates energy units according to algorithm
    :param del_q: integer - the change in energy
    :param func:
    :param i: particle idx
    :param solid: array representing the different particles
    """
    if del_q == -1 and solid[i] != 0:
        solid[i] -= 1  # remove 1 unit from particle
    elif del_q == 1:
        p = np.random.uniform(0, 1)
        if p <= func:
            solid[i] += 1


if __name__ == '__main__':
    # numeric_warmup(100, 100, 300, 0)
    # plt.savefig('q_a & q_b vs. number of iterations.jpg')
    # plt.show()
    metropolis_algo(theta=5)
