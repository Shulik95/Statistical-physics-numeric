import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import os


@njit
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


@njit
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


@njit
def check_converge(M_k, M_k_half, delta=10 ** -3):
    """

    :param M_k:
    :param M_k_half:
    :param delta:
    :return: return True if error converged false otherwise.
    """
    # print('Error is: ' + str({np.abs(M_k - M_k_half) / np.abs(M_k)}))
    err = np.abs(M_k - M_k_half) / np.abs(M_k)
    return err <= delta if M_k != 0 else False


@njit
def calc_heat_cap(T, U_tot, U_tot_sq, N):
    """
    :param T: float - system temperature
    :param U_tot: avg energy in system
    :param U_tot_sq: sq avg of energy in system
    :param N: number of particles in system
    :return: heat capacity relative to the given parameters.
    """
    return (1 / (N * (T ** 2))) * (U_tot_sq - U_tot ** 2)


@njit
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


@njit
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
    step_counter, M_k, U, U_sq, iter_counter, unchanged = 0, 0, 0, 0, 0, 0
    while step_counter < K and unchanged < 10000:
        curr_steps = scan_and_flip(grid, mu_B, J, T)

        # handles cases where grid doesnt change
        if curr_steps == 0:
            unchanged += 1
        elif curr_steps > 0:
            unchanged = 0
        step_counter += curr_steps
        iter_counter += 1
        if iter_counter % nsweep == 0:  # sample every 5th iteration
            grid_sum, U_tot = np.sum(grid), calc_U(grid, J, mu_B)
            M_k += grid_sum
            U += U_tot
            U_sq += U_tot ** 2

    M_k /= (iter_counter / nsweep)
    U /= (iter_counter / nsweep)
    U_sq /= (iter_counter / nsweep)
    return check_converge(M_k, M_k_half), M_k, U, U_sq, step_counter, iter_counter


@njit
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


@njit
def onsager(eta):
    """

    :param eta:
    :return:
    """
    z = np.exp(-2 * eta)
    mone = ((1 + z ** 2) ** (1 / 4)) * ((1 - 6 * z ** 2 + z ** 4) ** (1 / 8))
    mehane = np.sqrt((1 - z ** 2))
    return mone / mehane


def convergence(sqrt_N, eta_range1, h=0, magnetic=False):
    M_k_arr, U_arr, U_sq_arr, total_tries, tot_flips = [], [], [], [], []
    grid = np.random.choice([1, -1], (sqrt_N, sqrt_N))
    T = 1
    # B=0
    # plt.imshow(grid), plt.show()
    for eta in eta_range1:
        print("Starting eta = " + str(eta))
        K = 10000
        step_counter, iter_counter, nsamples, nsweep = 0, 0, 0, 5
        mu_B = h * T
        J = eta * T
        if eta == eta_range1[0]:
            M_k_half = first_iter(J, K, T, grid, iter_counter, mu_B, nsweep, step_counter)

        converged, M_k, U, U_sq = False, 0, 0, 0
        while not converged and K <= 10 ** 8:
            print(f"Not converged, Running with K ={K}\n")
            converged, M_k, U, U_sq, step_counter, iter_counter = K_iterations(T, K, mu_B, J, grid, M_k_half, nsweep)
            # plt.imshow(grid), plt.show()
            K *= 2
            M_k_half = M_k
        print("## Converged for K = " + str(K))
        M_k_arr.append(abs(M_k)), U_arr.append(U), U_sq_arr.append(U_sq), total_tries.append(
            iter_counter * sqrt_N ** 2), tot_flips.append(step_counter)
    if magnetic:
        return M_k_arr, U_arr
    return M_k_arr[:-1], U_arr[:-1], U_sq_arr[:-1], total_tries, tot_flips


def run_sim(sqrt_N=32):
    """

    :param sqrt_N:
    :return:
    """
    # create directory for plots
    current_dir = os.getcwd()
    final_dir = os.path.join(current_dir, rf'plots for N = {sqrt_N ** 2}')
    try:
        os.mkdir(final_dir)
    except OSError:
        print("Directory exists, no need to create a new one")

    eta_range1 = np.append(np.insert(np.delete(np.arange(0.1, 0.85, 0.05), 7), 7, np.arange(0.42, 0.46, 0.005)), 2)
    M_k_arr, U_arr, U_sq_arr, total_tries, tot_flips = convergence(sqrt_N, eta_range1)

    # plot magnetization
    plt.scatter(eta_range1[:-1], np.array(M_k_arr) / np.square(sqrt_N), label=r"$\frac{<M_k>}{N}$"), plt.title(
        r"$\frac{<M_k>}{N} vs. \eta$"), plt.xlabel(r"$\eta$"), plt.ylabel(
        r"$\frac{<M_k>}{N}$")
    plt.scatter(np.arange(0.4407, 0.8, 0.03), [onsager(eta) for eta in np.arange(0.4407, 0.8, 0.03)], marker='D',
                label="Onsager")
    plt.axvline(x=0.4406, ymin=0, linestyle='-.', color='k', label=r'$\eta_c$')
    plt.legend()
    plt.savefig(f"plots for N = {sqrt_N ** 2}/Magnetization vs eta.jpeg")
    plt.show()

    # plot energy
    plt.scatter(eta_range1[:-1], np.array(U_arr) / sqrt_N ** 2), plt.title(r"$\frac{<U>}{N} vs. \eta$"), plt.xlabel(
        r"$\eta$"), plt.ylabel(r"$\frac{<U>}{N}$")
    plt.savefig(f"plots for N = {sqrt_N ** 2}/Energy_vs_eta.jpeg")
    plt.show()

    # plot c_v
    plt.scatter(eta_range1[:-1], calc_heat_cap(1, np.array(U_arr), np.array(U_sq_arr), sqrt_N ** 2))
    plt.title(r"$c_v  vs. \eta$"), plt.xlabel(r"$\eta$"), plt.ylabel(r"$c_v$")
    plt.savefig(f'plots for N = {sqrt_N ** 2}/c_v vs eta.jpeg')
    plt.show()

    # plot ratio:
    plt.scatter(eta_range1, np.divide(np.array(tot_flips), np.array(total_tries)))
    plt.title(r"$Flip Ratio vs. \eta$"), plt.xlabel(r"$\eta$"), plt.ylabel(r"$Flip Ratio$")
    plt.savefig(f'plots for N = {sqrt_N ** 2}/Flip_Ratio.jpeg')
    plt.show()
    # B != 0

    eta_range2 = np.arange(0.1, 0.82, 0.05)
    h_range = [0, 0.1, 0.5, 1.0]
    for h in h_range:
        M_k_arr, U_arr = convergence(sqrt_N, eta_range2, h, True)

        # plot magnetization
        plt.subplot(121)
        plt.scatter(eta_range2, np.array(M_k_arr) / sqrt_N**2 , label=f'h={h}')

        # plot energy
        plt.subplot(122)
        plt.scatter(eta_range2, np.array(U_arr) / sqrt_N ** 2, label=f'h={h}')

    plt.subplot(121)
    plt.legend()
    plt.title(r"$\frac{<M_k>}{N} vs. \eta$"), plt.xlabel(
        r"$\eta$"), plt.ylabel(
        r"$\frac{<M_k>}{N}$")

    plt.subplot(122)
    plt.legend()
    plt.title(r"$\frac{<U>}{N} vs. \eta$"), plt.xlabel(
        r"$\eta$"), plt.ylabel(r"$\frac{<U>}{N}$")

    plt.tight_layout()
    plt.figure(figsize=(20, 20))
    plt.savefig(f"plots for N = {sqrt_N ** 2}/Magnetization(Energy)_vs_eta(h).jpeg")
    plt.show()


if __name__ == '__main__':
    run_sim()
