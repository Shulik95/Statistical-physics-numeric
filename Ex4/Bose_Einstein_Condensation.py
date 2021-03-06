import numpy as np
from matplotlib import pyplot as plt
from numba import njit

THREE_D = 3
n_MAX = 100
BASE_ENERGY = 0


@njit
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


@njit
def update_mu(T, mu_max, mu_min, n):
    """

    :param T: float - temperature
    :param mu_max: maximum chemical potential
    :param mu_min: minimum chemical potential
    :param n: number of energy levels
    :return: approximated number of particles, approximated chemical potential
    """
    mu_try = ((mu_max + mu_min) * 0.5)
    N_try = np.sum(np.array([get_degeneracy(3, i) * (1 / (np.exp((1 / T) * (i - mu_try)) - 1)) for i in range(n + 1)]))
    return N_try, mu_try


@njit
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
            mu_max = mu_try
            N_try, mu_try = update_mu(T, mu_try, mu_min, n)
        else:  # need to increase mu_try
            mu_min = mu_try
            N_try, mu_try = update_mu(T, mu_max, mu_try, n)
    return mu_try


def run_sim():
    """

    :return:
    """
    N_arr = [10]
    N_storage = [[],[],[],[]]
    N_sq_storage = [[],[],[],[]]
    cv_storage = [[],[],[],[]]

    for count, N in enumerate(N_arr):
        T_max = 25 if N == 10000 else 5 * np.log10(N)
        # N_storage[count] = np.append(N_storage[count], T_max)  # save T_max
        energy_arr = np.sort(np.concatenate([np.zeros(int((N * 0.7), )), np.random.randint(0, n_MAX + 1, size=(
            int(N * 0.3, )))]))  # init array with random energy levels.
        pi_n = np.cumsum(np.array([np.count_nonzero(energy_arr == n) / N for n in range(n_MAX + 1)]))
        for T in np.arange(0.2, T_max, 0.2):
            K = 50000
            print("*************************************")
            print(f"*** Starting T = {T} for N = {N} ***")
            print("*************************************")
            mu = find_chem_pot(N, T)
            for i in range(int(K / 2)):  # run until initial conditions are erased
                inner_iteration(N, T, mu, pi_n)
            print("done erasing initial conditions")
            # create <N_0>_K/2
            N0_k_half = 0
            for i in range(int(K / 2)):
                inner_iteration(N, T, mu, pi_n)
                N0_k_half += N * pi_n[0]
            N0_k_half = N0_k_half / K

            # create <N_0>_K
            N0_k, N0_sq = 0, 0
            K, N0_k, N0_sq = converge_iter(K, N, N0_k_half, N0_k, N0_sq, T, energy_arr, mu, pi_n)

            # save system
            N_storage[count].append(N0_k)
            N_sq_storage[count].append(N0_sq)
            U_tot, U_tot_sq = calc_tot_energy(energy_arr, n_MAX, N)
            cv_storage[count].append(calc_heat_cap(T, U_tot, U_tot_sq, N))

    return N_storage, N_sq_storage, cv_storage


def handle_edges(n, mu, T, p, pi_n, unit):
    """
    helper function, handles cases for highest energy/base level energy.
    """
    if n == 0:
        mone = get_degeneracy(THREE_D, n) / (np.exp((1 / T) * (- mu)) - 1)
        mehane = mone + get_degeneracy(THREE_D, n + 1) / (np.exp((1 / T) * (n + 1 - mu)) - 1)

        if p <= mone / mehane:  # stays at base energy
            return
        else:
            pi_n[n] -= unit  # receive one energy unit

    else:  # n == n_MAX
        mone = get_degeneracy(THREE_D, n - 1) / (np.exp((1 / T) * (n - 1 - mu)) - 1)
        mehane = + get_degeneracy(THREE_D, n) / (np.exp((1 / T) * (n + 1 - mu)) - 1)

        if p <= mone / mehane:
            pi_n[n - 1] += unit
        else:
            return


def inner_iteration(N, T, mu, pi_n):
    """
    simulates single iteration of system.
    :param N: int - number of particles.
    :param T: temperature of system.
    :param pi_n: array of cumulative probabilities
    :param mu: chemical potential of system.
    """
    p = np.random.uniform(0, 1)
    particle, i, unit = None, -1, 1 / N
    for n in range(len(pi_n)):
        low = pi_n[n - 1] if n != 0 else 0
        if low < p <= pi_n[n]:
            particle = n
            break

    # handle edges of array
    if particle in [0, 100]:
        handle_edges(particle, mu, T, p, pi_n, unit)
        return

    if p <= dec_energy_prob(particle, T, mu):  # particles give 1 energy unit to heat bath

        pi_n[particle - 1] += unit

    else:
        pi_n[particle] -= unit


def converge_iter(K, N, N0_avg, N0_k, N0_sq, T, energy_arr, mu, pi_n):
    """
    inner iteration for convergence.
    """
    while True:
        for i in range(K):
            inner_iteration(N, T, mu, pi_n)
            N0_k += N * pi_n[0]
            N0_sq += (N * pi_n[0]) ** 2
        N0_k /= K
        N0_sq /= K
        big_delta = get_delta(T)
        print(f"Convergance Error = {abs(N0_k - N0_avg) / N0_k}\n")
        if abs(N0_k - N0_avg) / N0_k <= big_delta:
            print("**********************************")
            print(f"*** converged after {K} steps ***")
            print("**********************************\n")
            break  # converged, leave loop
        else:  # didnt converge, edit parameters and run again
            N0_avg = N0_k
            print("*********************************************")
            print(f"*** Not converged - updating to K = {2 * K}***")
            print("*********************************************\n")
            K = 2 * K
    return K, N0_k, N0_sq


@njit
def dec_energy_prob(n, T, mu):
    """
    calculates the probability for a particle to give away a single energy unit
    :param T: temperature of system
    :param mu: chemical potential of system
    :param n: current number of energy units
    """
    mone = get_degeneracy(THREE_D, n - 1) / (np.exp((1 / T) * (n - 1 - mu)) - 1)
    mehane = mone + get_degeneracy(THREE_D, n + 1) / (np.exp((1 / T) * (n + 1 - mu)) - 1)
    return mone / mehane


@njit
def get_delta(T):
    """
    :param T: float - temperature in system
    :return: float - big delta for convergence criteria
    """
    big_delta = 10 ** -3
    if 1 < T <= 2:
        return 5 * big_delta
    else:
        return 10 ** -2


@njit
def calc_heat_cap(T, U_tot, U_tot_sq, N):
    """
    :param T: float - system temperature
    :param U_tot: avg energy in system
    :param U_tot_sq: sq avg of energy in system
    :param N: number of particles in system
    :return: heat capacity relative to the given parameters.
    """
    return (1 / N) * ((U_tot_sq / (T ** 2)) - (U_tot ** 2) / (T ** 2))


@njit
def calc_tot_energy(energy_arr, n_max, N):
    """
    calculates total energy avg
    :param energy_arr: np.array representing the energy of paricles
    :param n_max: int - maximum energy level possible
    :param N: number of particles in system
    """
    U_tot_avg = np.sum(np.array([n * (np.count_nonzero(energy_arr == n) / N) for n in range(n_max + 1)]))
    U_avg_sq = np.sum(np.array([(n ** 2) * (np.count_nonzero(energy_arr == n) / N) for n in range(n_max + 1)]))
    return U_tot_avg, U_avg_sq


if __name__ == '__main__':
    N0_storage, N0_sq_storage, cv_storage = run_sim()
    for idx, N in enumerate([10]):
        T_max = 25 if N == 10000 else 5 * np.log10(N)
        N0_arr = N0_storage[idx]
        N0_sq_arr = N0_sq_storage[idx]
        cv = cv_storage[idx]
        T_arr = np.arange(0.2, T_max, 0.2)
        Y_err = np.sqrt(np.array(N0_sq_arr) - np.square(np.array(N0_arr)))
        plt.scatter(T_arr, np.array(N0_arr)/N),
        plt.errorbar(T_arr, np.array(N0_arr)/N, yerr=Y_err, linestyle='None', color='r')
        temp = f"N = {N}: "
        plt.title(temp + r"$\frac{<N_0>}{N}$ vs. T"), plt.xlabel("Temperature"), plt.ylabel(
            r"$\frac{<N_0>}{N}$")
        plt.show()

        plt.scatter(T_arr, cv), plt.title(r"$C_v$ vs. T"), plt.xlabel("Temperature"), plt.ylabel(r"$C_v$")
        plt.show()
