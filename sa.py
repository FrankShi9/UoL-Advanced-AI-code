# simulated annealing algorithm
from matplotlib import pyplot
from numpy.ma import exp
from scipy import rand, randn



def simulated_annealing(objective_func, bounds, n_iter, step_size, temp):
    # random start
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # test start point
    best_eval = objective_func(best)

    # current feasible solution
    curr, curr_eval = best, best_eval
    scores = list()
    # run algorithm
    for i in range(n_iter):
        # step forward
        candidate = curr + randn(len(bounds)) * step_size
        # test candidate
        candidate_eval = objective_func(candidate)
        # update new best solution
        if candidate_eval < best_eval:  # loss < best
            # update
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            scores.append(best_eval)
            # output progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))

        # difference between candidate and current one
        diff = candidate_eval - curr_eval

        # calc temperature for current epoch
        t = temp / float(i + 1)

        # candidate metropolis acceptance criteria
        metropolis = exp(-diff / t)

        # check whether to keep the new one or not
        if diff < 0 or rand() < metropolis:
            # update new current point
            curr, curr_eval = candidate, candidate_eval

    # return [best, best_eval]
    return [best, best_eval, scores]

# objective function
def objective(x):
    return x[0] ** 2.0

from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed

# word output only
# # seed the pseudorandom number generator
# seed(1)
# # define range for input
# bounds = asarray([[-5.0, 5.0]])
# # define the total iterations
# n_iterations = 1000
# # define the maximum step size
# step_size = 0.1
# # initial temperature
# temp = 10
# # perform the simulated annealing search
# best, score = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
# print('Done!')
# print('f(%s) = %f' % (best, score))


# visualize
# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = asarray([[-5.0, 5.0]])
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# initial temperature
temp = 10
# perform the simulated annealing search
best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))
# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
