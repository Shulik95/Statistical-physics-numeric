import numpy as np
from matplotlib import pyplot as plt

THREE_D = 3
n_MAX = 100


def get_degeneracy(dim, n):
    """
    returns the degeneracy of the n-th energy level
    :param dim: the dimension of the problem.
    :param n: energy level.
    """
    if dim == THREE_D:
        return 0.5 * n * (n + 3) + 1
    else:
        return 1


def find_chem_pot(N, T, n=n_MAX):
    """

    :param N: int - number of particles.
    :param T: float - temperature
    :param n: number of energy states
    """
    mu_max, mu_min = 0, -30
    N_try, mu_try = update_mu(T, mu_max, mu_min, n)
    while int(N_try) != N:
        if N_try > N:  # need to decrease mu_try
            N_try, mu_try = update_mu(T, mu_try, mu_min, n)
        else:  # need to increase mu_try
            N_try, mu_try = update_mu(T, mu_max, mu_try, n)
    return mu_try


def update_mu(T, mu_max, mu_min, n):
    """

    :param T: float - temperature
    :param mu_max: maximum chemical potential
    :param mu_min: minimum chemical potential
    :param n: number of energy levels
    :return: approximated number of particles, approximated chemical potential
    """
    mu_try = int((mu_max + mu_min) * 0.5)
    N_try = np.sum(np.array([get_degeneracy(3, i) * (1 / (np.exp((1 / T) * (i - mu_try)) - 1)) for i in range(n + 1)]))
    return N_try, mu_try


def run_sim():
    """

    :return:
    """
    N_arr = [10, 100, 1000, 10000]
    for N in N_arr:
        T_max = 25 if N == 10000 else 5 * np.log10(N)
        K = N
        for T in np.arange(0.2, T_max, 0.2):
            mu = find_chem_pot(N, T)
            energy_arr = np.sort(
                np.random.randint(low=0, high=n_MAX + 1, size=(N,)))  # init array with random energy levels.
            for i in range(int(K / 2)):  # run until initial conditions are erased
                inner_iteration(N, T, energy_arr, mu)
            N0_avg = 0
            for i in range(int(K / 2)):
                N0_avg += inner_iteration(N, T, energy_arr, mu)
            N0_halfk = N0_avg / int(K/2)
            N0_k = 0
            for i in range(K):
                N0_k += inner_iteration(N,T,energy_arr,mu)
            N0_k /= K
            



def inner_iteration(N, T, energy_arr, mu):
    """
    simulates single iteration of system.
    :param N: int - number of particles.
    :param T: temperature of system.
    :param energy_arr: sorted array of particles and their energy levels.
    :param mu: chemical potential of system.
    """
    particle = np.randint(low=0, high=N)  # randomly choose particle
    energy_level = energy_arr[particle]
    p = np.random.uniform(low=0, high=1)
    if particle in [0, N - 1]:
        handle_edges()

    elif p <= dec_energy_prob(energy_level, T, mu):  # particles give 1 energy unit to heat bath
        energy_arr[particle] -= 1
    else:
        energy_arr[particle] += 1
    np.sort(energy_arr)
    return np.count_nonzero(energy_arr == 0)


def dec_energy_prob(n, T, mu):
    """
    calculates the probability for a particle to give away a single energy unit
    :param T: temperature of system
    :param mu: chemical potential of system
    :param n: current number of energy units
    """
    mone = get_degeneracy(THREE_D, n + 1) / (np.exp((1 / T) * (n + 1 - mu)) - 1)
    mehane = mone + get_degeneracy(THREE_D, n - 1) / (np.exp((1 / T) * (n - 1 - mu)) - 1)
    return mone / mehane


def handle_edges():
    pass
