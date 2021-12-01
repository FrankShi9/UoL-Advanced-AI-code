from pomegranate import BayesianNetwork
import seaborn, time, numpy, matplotlib
import pandas as pd
seaborn.set_style('whitegrid')



# dict with values as lists
X = pd.DataFrame({'A': [0, 0, 0, 1, 0], 'B': [0, 0, 1, 0, 0], 'C': [1, 1, 0, 0, 1], 'D': [0, 1, 0, 1, 1]})
X = X.to_numpy()

######################################################################
tic = time.time() #
# fit PGM from samples
model = BayesianNetwork.from_samples(X)
######################################################################

t = time.time() - tic
p = model.log_probability(X).sum()

print("Greedy")
print("Time (s): ", t)
# how does our model approximate actual distribution
print("P(D|M): ", p)
# model.plot()
######################################################################

tic = time.time()
model = BayesianNetwork.from_samples(X, algorithm='exact-dp')
t = time.time() - tic
p = model.log_probability(X).sum()
print("exact-dp")
print("Time (s): ", t)
print("P(D|M): ", p)
# model.plot()

######################################################################
tic = time.time()
model = BayesianNetwork.from_samples(X, algorithm='exact')
t = time.time() - tic
p = model.log_probability(X).sum()
print("exact")
print("Time (s): ", t)
print("P(D|M): ", p)
# model.plot()
######################################################################




