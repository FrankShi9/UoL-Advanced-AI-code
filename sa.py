# simulated annealing algorithm
from numpy.ma import exp
from scipy import rand, randn


def simulated_annealing(objective_func, bounds, n_iter, step_size, temp):
    # random start
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # test start point
    best_eval = objective_func(best)

    # current feasible solution
    curr, curr_eval = best, best_eval

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

    return [best, best_eval]

