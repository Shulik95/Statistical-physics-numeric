import numpy as np
from matplotlib import pyplot as plt


def calc_p(mu_B, J, grid, i, j, T):
    """

    :param mu_B:
    :param J:
    :param grid:
    :param i:
    :param j:
    :param T:
    :return:
    """
    sqrt_N = grid.shape[0]
    s_i = grid[i][j]
    temp = mu_B * s_i
    epsilon_now = -J * s_i * (
            grid[(i + 1) % sqrt_N, j] + grid[(i - 1) % sqrt_N][j] + grid[i][(j + 1) % sqrt_N] + grid[i][
        (j - 1) % sqrt_N]) - temp

    return 1 / (np.exp((-2 / T) * epsilon_now) + 1)


def scan_and_flip(grid, mu_B, J, T):
    """

    :param grid:
    :param mu_B:
    :param J:
    :param T:
    :return:
    """
    counter = 0
    sqrt_N = grid.shape[0]
    for i in range(sqrt_N):
        for j in range(sqrt_N):
            p_flip = calc_p(mu_B, J, grid, i, j, T)
            p = np.random.uniform(0, 1)
            if p <= p_flip:
                grid[i, j] *= -1
                counter += 1
    return counter


def check_converge(M_k, M_k_half, delta=10 ** -3):
    """

    :param M_k:
    :param M_k_half:
    :param delta:
    :return: return True if error converged false otherwise.
    """
    print(f'Error is: {np.abs(M_k - M_k_half) / np.abs(M_k)}')
    return np.abs(M_k - M_k_half) / np.abs(M_k) <= delta


def calc_heat_cap(T, U_tot, U_tot_sq, N):
    """
    :param T: float - system temperature
    :param U_tot: avg energy in system
    :param U_tot_sq: sq avg of energy in system
    :param N: number of particles in system
    :return: heat capacity relative to the given parameters.
    """
    return (1 / (N * (T ** 2))) * (U_tot_sq - U_tot ** 2)


def calc_U(grid, J, mu_B):
    """

    :param grid:
    :param J:
    :param mu_B:
    :return:
    """
    u_inter = 0
    sqrt_N = grid.shape[0]

    for i in range(sqrt_N):
        for j in range(sqrt_N):
            s_i = grid[i, j]
            u_inter += J * ((s_i * grid[i, (j + 1) % sqrt_N]) + (s_i * grid[(i + 1) % sqrt_N, j])) + mu_B * s_i
    return -1 * u_inter


def K_iterations(T, K, mu_B, J, grid, M_k_half, nsweep):
    """
    runs K iteration over the system, creating
    :param grid:
    :param T:
    :param K:
    :param eta:
    :param h:
    :return:
    """

    # create M_k
    step_counter, M_k, U, U_sq, iter_counter = 0, 0, 0, 0, 0
    while step_counter < K:
        step_counter += scan_and_flip(grid, mu_B, J, T)
        iter_counter += 1
        if iter_counter % nsweep == 0:  # sample every 5th iteration
            grid_sum, U_tot = np.sum(grid), calc_U(grid, J, mu_B)
            M_k += grid_sum
            U += U_tot
            U_sq += U_tot ** 2
    M_k /= (iter_counter / nsweep)
    U /= (iter_counter / nsweep)
    U_sq /= (iter_counter / nsweep)
    return check_converge(M_k, M_k_half), M_k, U, U_sq


def first_iter(J, K, T, grid, iter_counter, mu_B, nsweep, step_counter):
    """

    :param J:
    :param K:
    :param T:
    :param grid:
    :param iter_counter:
    :param mu_B:
    :param nsweep:
    :param step_counter:
    :return:
    """
    # forget initial conditions
    while step_counter < K // 2:
        step_counter += scan_and_flip(grid, mu_B, J, T)

    # create M_k/2
    M_k_half, step_counter = 0, 0
    while step_counter < K // 2:
        step_counter += scan_and_flip(grid, mu_B, J, T)
        iter_counter += 1
        if iter_counter % nsweep == 0:  # sample every 6th iteration
            M_k_half += np.sum(grid)
    M_k_half /= (iter_counter / nsweep)
    return M_k_half


def run_sim(N=32):
    eta_range = np.insert(np.delete(np.arange(0.1, 0.85, 0.05), 7), 7, np.arange(0.42, 0.46, 0.005))
    M_k_arr, U_arr, U_sq_arr = [], [], []
    grid = np.random.choice([1, -1], size=(N, N))
    T, h = 1, 0
    # B=0

    for eta in eta_range:
        K = 10000
        step_counter, iter_counter, nsamples, nsweep = 0, 0, 0, 5
        mu_B = h * T
        J = eta * T

        M_k_half = first_iter(J, K, T, grid, iter_counter, mu_B, nsweep, step_counter)

        converged, M_k, U, U_sq = False, 0, 0, 0
        while not converged and K <= 10 ** 8:
            print(f"Not converged, Running with K ={K}")
            converged, M_k, U, U_sq = K_iterations(T, K, mu_B, J, grid, M_k_half, nsweep)
            K *= 2
            M_k_half = M_k
        print(f"## Converged for K = {K} ##")
        M_k_arr.append(abs(M_k)), U_arr.append(U), U_sq_arr.append(U_sq)  # save converged state

    plt.scatter(eta_range, M_k_arr), plt.title(r"$<M_k> vs. \eta$"), plt.xlabel(r"$eta$"), plt.ylabel(r"$<M_k>$")
    plt.show()


if __name__ == '__main__':
    run_sim()
