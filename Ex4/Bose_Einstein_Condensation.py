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
    while N_try != N:
        if N_try > N:
            N_try, mu_try = update_mu(T, mu_try, mu_min, n)
        else:
            N_try, mu_try = update_mu(T, mu_max, mu_try, n)


def update_mu(T, mu_max, mu_min, n):
    """

    :param T:
    :param mu_max:
    :param mu_min:
    :param n:
    :return:
    """
    mu_try = int((mu_max + mu_min) * 0.5)
    N_try = np.sum(np.array([get_degeneracy(3, i) * (1 / (np.exp((1 / T) * (i - mu_try)) - 1)) for i in range(n + 1)]))
    return N_try, mu_try
