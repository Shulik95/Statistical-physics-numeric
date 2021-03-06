import math
import numpy as np
from numpy.linalg import norm
import matplotlib
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
        dtwall_x = (1 - self.rad - self.pos[0]) / vx if vx > 0 else (self.pos[0] - self.rad) / abs(vx)
        dtwall_y = (1 - self.rad - self.pos[1]) / vy if vy > 0 else (self.pos[1] - self.rad) / abs(vy)
        return min(dtwall_y, dtwall_x)

    def get_speed(self):
        """
        returns the speed of the current disc.
        """
        return norm(self.vel)


class Simulation:
    """simulation of circular particles in motion"""

    def __init__(self, v_table, p_table, nparticles=4, tot_v=2, dtstore=1, rad=0.15):
        self.v_table = v_table
        self.p_table = p_table
        self.rad = rad
        self.particles = [self.init_particles(i) for i in range(nparticles)]
        self.nparticles = nparticles
        self.tot_v = tot_v
        self.dtstore = dtstore
        self.t = 0
        self.stored = 1
        self.collision = 0
        self.storage = {0: [], 1: [], 2: [], 3: []}
        self.box2 = [[0 for k in range(10)] for p in range(10)]  # grid for 2nd particle graphs.
        self.box = [[0 for i in range(10)] for j in range(10)]  # grid to represent location.

    def init_particles(self, row):
        """
        return a new Particle object with position and velocity according
        to corresponding tables.
        """
        x, y = self.p_table[row]
        vx, vy = self.v_table[row]
        return Particle(x, y, vx, vy, rad=self.rad)

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
        temp = [(self.particles[i].time_to_wall(), i) for i in range(self.nparticles)]
        dtwall, p0 = min(temp, key=lambda t: t[0])

        min_dtcoll, p1, p2 = self.find_min_dtcoll()
        dt = min(dtwall, min_dtcoll)
        # print(dt)

        # update particle location
        for particle in self.particles:
            particle.advance(dt)

        # update time and velocity of colliding particles
        if self.t + dt > self.stored:
            self.stored += self.dtstore
            self.store_sim()

        self.t += dt
        if dtwall < min_dtcoll:
            self.update_vel_wall(p0)
            self.collision += 1
        else:
            self.update_vel_coll(p1, p2)
            self.collision += 1

    def store_sim(self):
        """
        stores the current vx,vy,v_abs of each particle
        :return:
        """
        for p in range(4):
            particle = self.particles[p]
            self.box[int(np.floor(particle.pos[0] * 10))][int(np.floor(particle.pos[1] * 10))] += 1  # update location
            self.storage[p].append([particle.vel[0], particle.vel[1], particle.get_speed()])  # store speeds
        particle = self.particles[1]
        self.box2[int(np.floor(particle.pos[0] * 10))][int(np.floor(particle.pos[1] * 10))] += 1  # update location
        #self.storage[1].append([particle.vel[0], particle.vel[1], particle.get_speed()])

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
        if self.particles[p0].pos[0] in [self.rad, 1-self.rad]:
            self.particles[p0].vel[0] *= -1
        else:
            self.particles[p0].vel[1] *= -1

    def update_vel_coll(self, p1, p2):
        """
        updates velocity for particles after collision.
        :param p1: idx of 1st particle
        :param p2: idx of 2nd particle
        """
        delta_pos = self.particles[p2].pos - self.particles[p1].pos
        delta_vel = self.particles[p2].vel - self.particles[p1].vel
        deltal = norm(delta_pos)
        # deltav = norm(delta_vel)
        ex = delta_pos[0] / deltal
        ey = delta_pos[1] / deltal
        s_v = delta_vel[0] * ex + delta_vel[1] * ey
        self.particles[p1].vel[0] = self.particles[p1].vel[0] + ex * s_v
        self.particles[p1].vel[1] = self.particles[p1].vel[1] + ey * s_v
        self.particles[p2].vel[0] = self.particles[p2].vel[0] - ex * s_v
        self.particles[p2].vel[1] = self.particles[p2].vel[1] - ey * s_v


def __heatmap(data, row_labels, col_labels, ax=None,
              cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == '__main__':
    # Useful definitions
    p_table = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    v_table = np.array([[0.21, 0.12], [0.71, 0.18], [-0.23, -0.79], [0.78, 0.34583]])
    v_hist_bins = np.linspace(-np.sqrt(2), np.sqrt(2), 200)
    v_abs_bins = np.linspace(0, np.sqrt(2), 100)
    pos_arr = []
    legend_arr = [r"$v_x$", r"$v_y$"]

    # comparison graphs plot
    for r in np.arange(0.1, 0.24, 0.01):
        sim1 = Simulation(v_table, p_table, rad=r)
        counter = 0
        while sim1.collision < 10 ** 7:
            sim1.advance()
            print(sim1.collision)
        in_range_percentage = np.sum(np.array(sim1.box2)[0:5, 0:5]) / sim1.stored
        pos_arr.append(in_range_percentage)
        p2_storage = sim1.storage[1]
        plt.hist(np.array(p2_storage).T[0], v_hist_bins, density=True, histtype='step', label='r = ' + str(r))
        print("######## " + str(r) + " ############")

    plt.title(r"$P(v_x)$ comparison"), plt.xlabel(r"$v_x$"), plt.ylabel(r'$P(v_x)$'), plt.legend()
    plt.savefig("velocity sensitivity for r "), plt.show()

    plt.scatter(np.arange(0.1, 0.24, 0.01), pos_arr), plt.ylabel(r"% in $1^{st}$ quarter"), plt.xlabel(
        "Radius"), plt.title("Location sensitivity vs. Radius")
    plt.savefig("Location sensitivity vs Radius")
    plt.show()

    # first part graphs
    sim = Simulation(v_table, p_table)
    counter = 0
    while sim.collision < 10 ** 6:
        sim.advance()

    # plot heat map
    im, bins = __heatmap(np.array(sim.box) / sim.stored, np.around(np.linspace(0, 1, 10), 1),
                         np.around(np.linspace(0, 1, 10), 1))
    plt.title("")
    plt.tight_layout()
    plt.show()

    # plot velocity probability

    p1_arr, p2_arr, p3_arr, p4_arr = sim.storage[0], sim.storage[1], sim.storage[2], sim.storage[3]

    for i in range(2):
        for j, elem in enumerate([p1_arr, p2_arr, p3_arr, p4_arr]):
            plt.hist(np.array(elem).T[i], v_hist_bins, density=True, histtype='step', label=legend_arr[i] + str(j))
        plt.legend()
        plt.title(legend_arr[i] + " probability density"), plt.xlabel(legend_arr[i]), plt.ylabel("Density")
        plt.savefig(legend_arr[i] + ".jpeg")
        plt.show()

    for idx, elem in enumerate([p1_arr, p2_arr, p3_arr, p4_arr]):
        plt.hist(np.array(elem).T[2], v_abs_bins, density=True, histtype='step', label=r'$v_abs$ ' + str(idx))
    plt.legend()
    plt.title(r"$v_{abs}$ probability density"), plt.xlabel(r"$v_{abs}$"), plt.ylabel("Density")
    plt.savefig("v_abs.jpeg")
    plt.show()
