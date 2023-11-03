from math import floor
from matplotlib import pyplot as plt
import numpy as np
import time


class Geno_TSP(object):

    def __init__(self, data,
                 max_gen=1000,
                 pop_size=100,
                 cross_proba=0.80,
                 mutate_proba=0.02,
                 survive_proba=0.8):
        # max iteration num
        self.max_gen = max_gen
        # population quantity
        self.pop_size = pop_size
        # crossover probability
        self.cross_proba = cross_proba
        # mutate probability
        self.mutate_proba = mutate_proba
        # survive probability
        self.survive_proba = survive_proba
        # city location data, 2d-array
        self.data = data
        # city number, important for encoding,
        # using integral encoding,
        # also corresponding to chromosome length
        self.city_num = len(data)
        # distance matrix
        self.matrix_distance = self.matrix_dis()
        # survive number for children generation
        self.survive_num = max(floor(self.pop_size * self.survive_proba + 0.5), 2)
        # initialization of father and child generation
        self.father_chrom = np.zeros((self.pop_size, self.city_num), dtype=np.int8)
        self.child_chrom = np.zeros((self.survive_num, self.city_num), dtype=np.int8)
        # fitness function, here will use reciprocal of distance
        self.fitness = np.zeros(self.pop_size)
        # the optimized fitness value
        self.opt_fit = []
        # the optimized path, which is the output result
        self.opt_path = []

    def matrix_dis(self):
        """
            calculate distance matrix of all input cities
            ---------------------------------------------------------------
            :parameter
            None
        :return:
            {np.ndarray}
            distance matrix
        """
        res = np.zeros((self.city_num, self.city_num))
        for i in range(self.city_num):
            for j in range(i + 1, self.city_num):
                res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
                res[j, i] = res[i, j]
        return res

    def randomize_chrom(self):
        """
            generate initial population using random methods
            ------------------------------------------------------------
            will tweak father_chrom, no returns
        :return:
        """
        # got one permuted single chromosome
        rand_single_chrome = np.array(range(self.city_num))
        for i in range(self.pop_size):
            np.random.permutation(rand_single_chrome)
            self.father_chrom[i, :] = rand_single_chrome
            self.fitness[i] = self.compute_fitness(rand_single_chrome)

    def compute_fitness(self, rand_single_path):
        """
            calculate path distance, then transfer to fitness using reciprocal
            as survival function. note that encoding order is exactly the path
            order
            ------------------------------------------------------------------
        :param
            rand_single_path: {np 1d array}
            one order series of single chromosome
        :return:
            res:{np float64}
            fitness value
        """
        res = 0
        for i in range(self.city_num - 1):
            res += self.matrix_distance[rand_single_path[i], rand_single_path[i + 1]]
        res += self.matrix_distance[rand_single_path[-1], rand_single_path[0]]
        return res

    def out_path(self, one_path):
        """
            visualization of path
            -------------------------------------------------
        :param
            one_path: {np 1d array}
            one order series of single chromosome
        :return:
            None
        """
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.city_num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)

    def select_child(self):
        """
            to choose chromosomes whose fitness function value is higher than the average
            also, numbers of child generation is limited by self.survive_num
            this is done by random choosing process
            ----------------------------------------------
        :parameter
            None
        :return:
            None
        """

        # determine fitness function
        fit = 1.0 / self.fitness
        # calculate overall fitness value
        cumsum_fit = np.cumsum(fit)
        thresh_fit = cumsum_fit[-1] / self.survive_num * (np.random.rand() + np.array(range(int(self.survive_num))))
        # start selecting process
        i, j = 0, 0
        index = []
        while i < self.pop_size and j < self.survive_num:
            if cumsum_fit[i] >= thresh_fit[j]:
                index.append(i)
                j += 1
            else:
                i += 1
        self.child_chrom = self.father_chrom[index, :]

    def crossover(self):
        """
            crossover process
        :return:
        """
        # first decide the total num of child generation is odd or even
        # for the purpose that selecting chromosomes in pair
        if self.survive_num % 2 == 0:
            ttl_num = range(0, int(self.survive_num), 2)
        else:
            ttl_num = range(0, int(self.survive_num - 1), 2)
        for i in ttl_num:
            if self.cross_proba >= np.random.rand():
                self.child_chrom[i, :], self.child_chrom[i + 1, :] \
                    = self.intercross(self.child_chrom[i, :], self.child_chrom[i + 1, :])

    def intercross(self, chrom_a, chrom_b):
        """
            do inter-cross
            basic concept is to find two positions in both chromosomes
            then exchange values located between these two positions
            due to the fact that our encoding method is integral,
            we need also to check the validity of chromosome after
            each crossover, as no duplicated numbers would appear in
            the same chromosome
        :param chrom_a:
        :param chrom_b:
        :return:
            inter-crossed chromosomes
        """
        # find two positions in chromosome
        p1 = np.random.randint(self.city_num)
        p2 = np.random.randint(self.city_num)
        while p1 == p2:
            p2 = np.random.randint(self.city_num)
        p_left, p_right = min(p1, p2), max(p1, p2)
        # prepare two parental chromosome
        ind_a1 = chrom_a.copy()
        ind_b1 = chrom_b.copy()
        for i in range(p_left, p_right + 1):
            # this copy is for validity check
            ind_a2 = chrom_a.copy()
            ind_b2 = chrom_b.copy()
            chrom_a[i] = ind_b1[i]
            chrom_b[i] = ind_a1[i]
            # x,y will be a np array containing index of True return of mask
            x = np.argwhere(chrom_a == chrom_a[i])
            y = np.argwhere(chrom_b == chrom_b[i])
            if len(x) == 2:
                chrom_a[x[x != i]] = ind_a2[i]
            if len(y) == 2:
                chrom_b[y[y != i]] = ind_b2[i]
        return chrom_a, chrom_b

    def mutate(self):
        """
            mutate
            mutation here is randomly interchange values in two positions
        :return:
        """
        for i in range(self.survive_num):
            # do mutation while random number pass preset mutation probability
            if np.random.rand() <= self.mutate_proba:
                m1 = np.random.randint(self.city_num)
                m2 = np.random.randint(self.city_num)
                while m1 == m2:
                    m2 = np.random.randint(self.city_num)
                self.child_chrom[i, [m1, m2]] = self.child_chrom[i, [m2, m1]]

    def reverse(self):
        """
            do reverse process
        :return:
        """
        for i in range(int(self.survive_num)):
            r1 = np.random.randint(self.city_num)
            r2 = np.random.randint(self.city_num)
            while r1 == r2:
                r2 = np.random.randint(self.city_num)
            r_left, r_right = min(r1, r2), max(r1, r2)
            temp_chrom = self.child_chrom[i, :].copy()
            temp_chrom[r_left:r_right + 1] = self.child_chrom[i, r_left:r_right + 1][::-1]
            # if fitness doesn't increase, restore child chromosome
            if self.compute_fitness(temp_chrom) < self.compute_fitness(self.child_chrom[i, :]):
                self.child_chrom[i, :] = temp_chrom

    def replace(self):
        """
            replace parent chromosome with lower fitness value with child chromosome
        :return:
        """
        index = np.argsort(self.fitness)[::-1]
        self.father_chrom[index[:len(self.child_chrom)], :] = self.child_chrom

    def ori_plot(self):
        """
            plot the very first population quantity
            note that father_generation should be initialized first
            run after randomize_chrom() function
            -----------------------------------------------
        :return:
        """
        fig, ax = plt.subplots()
        x = self.data[:, 0]
        y = self.data[:, 1]
        ax.scatter(x, y, lw=0.1, marker='x')
        for i, s in enumerate(range(1, len(self.data) + 1)):
            ax.annotate(s, (x[i], y[i]))
        res0 = self.father_chrom[0]
        x0 = x[res0]
        y0 = y[res0]
        for i in range(len(self.data) - 1):
            plt.quiver(x0[i], y0[i], x0[i + 1] - x0[i], y0[i + 1] - y0[i],
                       color='r', lw=0.005, angles='xy', scale=1, scale_units='xy')
        plt.quiver(x0[-1], y0[-1], x0[0] - x0[-1], y0[0] - y0[-1],
                   color='r', lw=0.005, angles='xy', scale=1, scale_units='xy')
        plt.show()

    def run(self):
        """
            start algorithm
        :return:
        """
        self.randomize_chrom()
        self.ori_plot()
        print("original chromosome path: " + str(self.fitness[0]))
        for i in range(self.max_gen):
            self.select_child()
            self.crossover()
            self.mutate()
            self.reverse()
            self.replace()
            # calculate new fitness value
            for j in range(self.pop_size):
                self.fitness[j] = self.compute_fitness(self.father_chrom[j, :])
            # output optimal path in every 25 iteration
            index = self.fitness.argmin()
            if (i + 1) % 25 == 0:
                timestamp = time.time()
                formatted_time = time.strftime("%Y-%m-%d %H:%H:%S", time.localtime(timestamp))
                print(formatted_time)
                print(str(i + 1) + "th generation's optimal path length is: " + str(self.fitness[index]))
                print(str(i + 1) + "th generation's optimal path routine is: ")
                self.out_path(self.father_chrom[index, :])

            # store optimal fitness and chromosome series
            self.opt_fit.append(self.fitness[index])
            self.opt_path.append(self.father_chrom[index, :])

        # plot the optimal path
        self.ori_plot()
        return self


data = np.array([20.00, 96.10, 16.47, 94.44,
                 20.09, 92.54, 22.39, 93.37,
                 25.23, 97.24, 22.00, 96.05,
                 20.47, 97.02, 17.20, 96.29,
                 16.30, 97.38, 25.05, 98.12,
                 16.53, 96.50, 21.52, 95.59,
                 19.41, 97.13, 20.09, 92.55,
                 20.10, 95.10, 20.20, 94.10,
                 20.60, 96.88, 18.66, 90.20]).reshape(18, 2)

if __name__ == "__main__":
    geno_algo = Geno_TSP(data)
    geno_algo.run()
