import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann

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
    N0_dict = {i: [] for i in N_arr}
    N0_sq_dict = {j: [] for j in N_arr}
    cv_dict = {k: [] for k in N_arr}

    for N in N_arr:
        T_max = 25 if N == 10000 else 5 * np.log10(N)
        N0_dict[N].append(T_max)
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
            N0_avg = N0_avg / int(K / 2)
            N0_k, N0_sq = 0, 0
            while True:
                for i in range(K):
                    val = inner_iteration(N, T, energy_arr, mu)
                    N0_k += val * (val / N)
                    N0_sq += (val / N) * (val ** 2)
                # N0_k /= K
                # N0_sq /= K
                big_delta = get_delta(T)
                if abs(N0_k - N0_avg) / N0_k <= big_delta:
                    print(f"*** converged after {K} steps ***")
                    break  # converged, leave loop
                else:  # didnt converge, edit parameters and run again
                    N0_avg = N0_k
                    K = 2 * K

            # save system
            N0_dict[N].append(N0_k / N)
            N0_sq_dict[N].append(np.sqrt(N0_sq - N0_k ** 2) / N0_k)
            U_tot, U_tot_sq = calc_tot_energy(energy_arr, n_MAX, N)
            cv_dict[N].append(calc_heat_cap(T, U_tot, U_tot_sq, N))
    return N0_dict, N0_sq_dict, cv_dict


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
    mone = get_degeneracy(THREE_D, n - 1) / (np.exp((1 / T) * (n - 1 - mu)) - 1)
    mehane = mone + get_degeneracy(THREE_D, n - 1) / (np.exp((1 / T) * (n - 1 - mu)) - 1)
    return mone / mehane


def handle_edges(idx, n, T, mu):
    # TODO: handle edge cases
    



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


def calc_heat_cap(T, U_tot, U_tot_sq, N):
    """
    :param T: float - system temperature
    :param U_tot: avg energy in system
    :param U_tot_sq: sq avg of energy in system
    :param N: number of particles in system
    :return: heat capacity relative to the given parameters.
    """
    return (1 / N) * ((U_tot_sq / (T ** 2)) - (U_tot ** 2) / (T ** 2))


def calc_tot_energy(energy_arr, n_max, N):
    """
    calculates total energy avg
    :param energy_arr: np.array representing the energy of paricles
    :param n_max: int - maximum energy level possible
    :param N: number of particles in system
    """
    U_tot_avg = sum([n * (np.count_nonzero(energy_arr == n) / N) for n in range(n_max + 1)])
    U_avg_sq = sum([(n ** 2) * (np.count_nonzero(energy_arr == n) / N) for n in range(n_max + 1)])
    return U_tot_avg, U_avg_sq


if __name__ == '__main__':
    N0_dict, N0_sq_dict, cv_dict = run_sim()
    for N in [10, 100, 1000, 10000]:
        T_max = N0_dict[N][0]
        N0_arr = N0_dict[N][1:]
        N0_sq_arr = N0_sq_dict[N]
        cv = cv_dict[N]
        T_arr = np.arange(0.2, T_max, 0.2)

        plt.plot(T_arr, N0_arr), plt.title(r"$\frac{<N_0>}{N}$ vs. T"), plt.xlabel("Temperature"), plt.ylabel(
            r"$\frac{<N_0>}{N}$")
        plt.show()

        plt.plot(T_arr, cv), plt.title(r"$C_v$ vs. T"), plt.xlabel("Temperature"), plt.ylabel(r"$C_v$")
        plt.show()
