# Code Explanation
## 1 Overview

5 parts are included in Genetic Algorithm[遗传算法]. These are

​	> encoding strategy

​	> fitness function 

​	> mutation operator

​	> cross-over operator

​	> selection operator

It's noteworthy that three operators are most important component of a successful genetic algorithm

The general procedure of a genetic algorithm should at least includes following steps:

1. Describing question and encode population
2. Initialize starting population
3. Apply selection operator according to fitness function to determine child population
4. Apply cross-over operator. This is the most important part of the whole algorithm, which would drastically influence convergence rate
5. Apply mutation operator. This operator will introduce new gene types into exsiting population, mathmatical-wise it will help the whole system avoid running into local extremum.
6. Iterate  the whole process untill convergence.

## 2 Code Review

### 2.1 Encoding Strategy

For encoding strategy, the crucial part is that transfer features of problem into a series of codes. Note that the processing object of GA is a series of codes rather meaningful numbers or matrix. There're at least 4 popular genres:

1. binary encoding
2. gray code encoding
3. Float number encoding
4. Symbolic encoding

In our code, TSP problem intrinsicly allows us to put city numer as code, thus we just simply use integer as encoded series.

### 2.2 Fitness Function

We simply use cost function as fitness function. In TSP problem, cost function is the sum of distance through routine. You could find this in compute_fitness() function

```code
    def compute_fitness(self, rand_single_path):
        res = 0
        for i in range(self.city_num - 1):
            res += self.matrix_distance[rand_single_path[i], rand_single_path[i + 1]]
        res += self.matrix_distance[rand_single_path[-1], rand_single_path[0]]
        return res
```

### 2.3 Selection Function

Several methodologies are introduced in this section, generally, we could narrow down them into 4 different genre:

1.  Proportional Model, randomly choose unit from parent generation
2. Retain Optimal Model, this would take two steps, first find several optimal unit by setting certain thresholds, then retain them directly to the child generation, then apply other operators, replace units with relatively lower fitness function values with newly generated units. We generally apply this in our code.

```code
    def select_child(self):
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
        
    def replace(self):
        index = np.argsort(self.fitness)[::-1]
        self.father_chrom[index[:len(self.child_chrom)], :] = self.child_chrom
```

3. Random choose Model, pick units with hight expectation value
4. Sort Model, first sort population in descending order according to its fitness value, then assign probability to each unit according to its order, then ally proportional model to this population.

### 2.4 Operators

The basic concept for cross-over, mutation and reverse operators are as follows:

1. first find exact position in assigned gene or gene pairs
2. then apply operations to genetic series 
3. last check validity of transferred gene series

It's quite straightforward to understand concepts in code. These functions are:

​	> crossover()

​	> mutate()

​	> reverse()

### 2.5 Iteration

All in run() function. In this case, we show result in every 25 epochs, also out_path() and ori_plot() serve as printing path and showing path in figure.



## 3 Input and Parameters

### 3.1 Inputs

In TSP problem, we only take city position, which is one 2d array as input. 

Our code will calculate distance matrix as well as generate initial. encoded population automatically. 

```code
self.city_num = len(data)
# distance matrix
self.matrix_distance = self.matrix_dis()

def matrix_dis(self):
	res = np.zeros((self.city_num, self.city_num))
	for i in range(self.city_num):
		for j in range(i + 1, self.city_num):
			res[i, j] = np.linalg.norm(self.data[i, :] - self.data[j, :])
			res[j, i] = res[i, j]
	return res
```

### 3.2 Parameters

```code
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
```

