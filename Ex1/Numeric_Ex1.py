import numpy as np
from matplotlib import pyplot as plt


def numeric_warmup(N_a, N_b, q_a, q_b):
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
            E_a = sum(q_arr[:N_a])
            E_b = sum((q_arr[N_a:]))
            if i < 100 <= j:  # from solid A to B
                if E_b > E_a:
                    counter += 1
            elif j < 100 <= i:
                if E_a > E_b:
                    counter += 1
            q_arr[i] -= 1
            q_arr[j] += 1
        q_a_arr.append(sum(q_arr[:N_a]))
        q_b_arr.append(sum(q_arr[N_a:]))

    # plot data
    plt.plot(range(10 ** 5), q_a_arr, 'r', label="Total energy in solid A")
    plt.plot(range(10 ** 5), q_b_arr, 'b', label="Total energy in solid B")
    plt.title(r'$q_a & q_b$ vs. # of iterations')
    plt.legend(), plt.grid()
    print(counter)


if __name__ == '__main__':
    numeric_warmup(100, 100, 300, 0)
    # plt.savefig('q_a & q_b vs. number of iterations.jpg')
    plt.show()
