from pomegranate import *

# # monty-hall problem
# guest = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})
# prize = DiscreteDistribution({'A': 1. / 3, 'B': 1. / 3, 'C': 1. / 3})
# monty = ConditionalProbabilityTable(
#     [['A', 'A', 'A', 0.0],
#      ['A', 'A', 'B', 0.5],
#      ['A', 'A', 'C', 0.5],
#      ['A', 'B', 'A', 0.0],
#      ['A', 'B', 'B', 0.0],
#      ['A', 'B', 'C', 1.0],
#      ['A', 'C', 'A', 0.0],
#      ['A', 'C', 'B', 1.0],
#      ['A', 'C', 'C', 0.0],
#      ['B', 'A', 'A', 0.0],
#      ['B', 'A', 'B', 0.0],
#      ['B', 'A', 'C', 1.0],
#      ['B', 'B', 'A', 0.5],
#      ['B', 'B', 'B', 0.0],
#      ['B', 'B', 'C', 0.5],
#      ['B', 'C', 'A', 1.0],
#      ['B', 'C', 'B', 0.0],
#      ['B', 'C', 'C', 0.0],
#      ['C', 'A', 'A', 0.0],
#      ['C', 'A', 'B', 1.0],
#      ['C', 'A', 'C', 0.0],
#      ['C', 'B', 'A', 1.0],
#      ['C', 'B', 'B', 0.0],
#      ['C', 'B', 'C', 0.0],
#      ['C', 'C', 'A', 0.5],
#      ['C', 'C', 'B', 0.5],
#      ['C', 'C', 'C', 0.0]], [guest, prize])
#
# s1 = Node(guest, name="guest")
# s2 = Node(prize, name="prize")
# s3 = Node(monty, name="monty")
#
# model = BayesianNetwork("Monty Hall Problem")
# model.add_states(s1, s2, s3)
# model.add_edge(s1, s3)
# model.add_edge(s2, s3)
# model.bake()
#
# print(model.probability([['A', 'A', 'A'],
#                          ['A', 'A', 'B'],
#                          ['C', 'C', 'B']]))
# print('-----------------------------------------------')
# print(model.predict([['A', 'A', None],
#                      ['A', 'A', None],
#                      ['C', 'C', None]]))
# print('-----------------------------------------------')
# print(model.predict_proba([['A', 'A', None],
#                            ['A', 'A', None],
#                            ['C', 'C', None]]))
# print('-----------------------------------------------')
# print(model.predict_proba([['B', None, None]]))
# print('-----------------------------------------------')
# print('model.marginal ', model.marginal())
#
# print('log_probability ', model.log_probability([['A', 'A', 'A']]))
# # end of monty hall


# model of example from TB cpt 6.1
camera = DiscreteDistribution({'0': 0.6, '1': 0.4})
radar = DiscreteDistribution({'0': 0.5, '1': 0.5})
fog = ConditionalProbabilityTable(
        [['0', '0', 0.5],
         ['0', '1', 0.5],
         ['1', '0', 0.3],
         ['1', '1', 0.7]], [camera])
away = ConditionalProbabilityTable(
        [['0', '0', 0.2],
         ['0', '1', 0.8],
         ['1', '0', 0.6],
         ['1', '1', 0.4]], [radar])
detected = ConditionalProbabilityTable(
        [['0', '0', '0', 1.0],
         ['0', '0', '1', 0.0],
         ['0', '1', '0', 0.6],
         ['0', '1', '1', 0.4],
         ['1', '0', '0', 0.7],
         ['1', '0', '1', 0.3],
         ['1', '1', '0', 0.1],
         ['1', '1', '1', 0.9]], [camera, radar])
stopped = ConditionalProbabilityTable(
        [['0', '0', '0', 0.6],
         ['0', '0', '1', 0.4],
         ['0', '1', '0', 0.9],
         ['0', '1', '1', 0.1],
         ['1', '0', '0', 0.1],
         ['1', '0', '1', 0.9],
         ['1', '1', '0', 0.5],
         ['1', '1', '1', 0.5]], [detected, away])

s1 = Node(camera, name="camera")
s2 = Node(radar, name="radar")
s3 = Node(fog, name="fog")
s4 = Node(away, name="away")
s5 = Node(detected, name="detected")
s6 = Node(stopped, name="stopped")


model = BayesianNetwork("Pedestrian Detection Problem")
model.add_states(s1, s2, s3, s4, s5, s6)
model.add_edge(s1, s3)
model.add_edge(s1, s5)
model.add_edge(s2, s4)
model.add_edge(s2, s5)
model.add_edge(s5, s6)
model.add_edge(s4, s6)
model.bake()
# end of model


# # causal reasoning example from TB cpt 6.1

# #### P(s0)
# query = [None, None, None, None, None, '0']
# ps0 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 for j5 in range(2):
#                     ps0 += model.probability([[str(j1), str(j2), str(j3), str(j4), str(j5), '0']])
# print("the probability of the car does not stop P(s0): %s\n"%ps0)
#
# #### P(s0,c1)
# query = ['1', None, None, None, None, '0']
# ps0c1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 ps0c1 += model.probability([['1', str(j1), str(j2), str(j3), str(j4), '0']])
# print("the probability of the car does not stop but the camera captured the pedestrian P(s0,c1): %s\n"%(ps0c1))
#
# ### P(c1)
# query = ['1', None, None, None, None, None]
# pc1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 for j5 in range(2):
#                     pc1 += model.probability([['1', str(j1), str(j2), str(j3), str(j4), str(j5)]])
# print("the probability of camera captured the pedestrian P(c1): %s\n"%pc1)
#
#
# #### P(s0|c1)
# print("the conditional probability of the car does not stop when the camera captured the pedestrian P(s0|c1)): %s\n"%(ps0c1/pc1))
#
#
# #### P(s0,r1)
# query = [None, '1', None, None, None, '0']
# ps0r1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 ps0r1 += model.probability([[str(j1), '1', str(j2), str(j3), str(j4), '0']])
# print("the probability of the car does not stop but the radar captured the pedestrian P(s0,r1): %s\n"%(ps0r1))
#
# ### P(r1)
# query = [None, '1', None, None, None, None]
# pr1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 for j5 in range(2):
#                     pr1 += model.probability([[str(j1), '1', str(j2), str(j3), str(j4), str(j5)]])
# print("the probability of radar captured the pedestrian P(r1): %s\n"%pr1)
#
# #### P(s0|r1)
# print("the conditional probability of the car does not stop when the radar captured the pedestrian P(s0|r1): %s\n"%(ps0r1/pr1))
#
#
# #### P(s0,c1,r1)
# query = ['1', '1', None, None, None, '0']
# ps0c1r1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#                 ps0c1r1 += model.probability([['1', '1', str(j1), str(j2), str(j3), '0']])
# print("the probability of the car does not stop but both the camera and the radar captured the pedestrian P(s0,c1,r1): %s\n"%(ps0c1r1))
#
# #### P(c1,r1)
# query = ['1', '1', None, None, None]
# pc1r1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 pc1r1 += model.probability([['1', '1', str(j1), str(j2), str(j3), str(j4)]])
# print("the probability of both the camera and the radar captured the pedestrian  P(c1,r1): %s\n"%(pc1r1))
#
#
# #### P(s0|c1,r1)
# print("the conditional probability of the car does not stop when the camera captured the pedestrian P(s0|c1,r1): %s\n"%(ps0c1r1/pc1r1))
#
# # end of causal reasoning

## evidental reasoning example from TB cpt 6.1
# ### P(c1)
# query = ['1', None, None, None, None, None]
# pc1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 for j5 in range(2):
#                     pc1 += model.probability([['1', str(j1), str(j2), str(j3), str(j4), str(j5)]])
# print("the probability of camera captured the pedestrian P(c1): %s\n"%pc1)
#
# #### P(c1,s1)
# query = ['1', None, None, None, None, '1']
# pc1s1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 pc1s1 += model.probability([['1', str(j1), str(j2), str(j3), str(j4), '1']])
# print("the probability of the car stopped and the camera captured the pedestrian P(c1,s1): %s\n"%(pc1s1))
#
#
# #### P(s1)
# query = [None, None, None, None, None, '1']
# ps1 = 0
# for j1 in range(2):
#     for j2 in range(2):
#         for j3 in range(2):
#             for j4 in range(2):
#                 for j5 in range(2):
#                     ps1 += model.probability([[str(j1), str(j2), str(j3), str(j4), str(j5), '1']])
# print("the probability of the car stopped P(s1): %s\n"%ps1)
#
#
# #### P(c1|s1)
# print("the conditional probability of the camera captured the pedestrian when seeing that the car stopped P(c1|s1): %s\n"%(pc1s1/ps1))

## end of evidental reasoning

# intercausal reasoning
### P(r1)
query = [None, '1', None, None, None, None]
pr1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                for j5 in range(2):
                    pr1 += model.probability([[str(j1), '1', str(j2), str(j3), str(j4), str(j5)]])
print("P(r1): %s\n"%pr1)

### P(d1)
query = [None, None, None, None, '1', None]
pd1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                for j5 in range(2):
                    pd1 += model.probability([[str(j1), str(j2), str(j3), str(j4), '1', str(j5)]])
print("P(d1): %s\n"%pd1)

#### P(r1,d1)
query = [None, '1', None, None, '1',  None]
pr1d1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                pr1d1 += model.probability([[str(j1), '1', str(j2), str(j3), '1', str(j4)]])
print("P(r1,d1): %s\n"%(pr1d1))

#### P(r1|d1)
print("P(r1|d1)): %s\n"%(pr1d1/pd1))

#### P(r1,d1,c1)
query = ['1', '1', None, None, '1',  None]
pr1d1c1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            pr1d1c1 += model.probability([['1',  '1', str(j2), str(j1),'1', str(j3)]])
print("P(r1,d1,c1): %s\n"%(pr1d1c1))

#### P(d1,c1)
query = ['1', None, None, None,  '1',  None]
pd1c1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                pd1c1 += model.probability([['1',  str(j2), str(j3), str(j1),'1', str(j4)]])
print("P(d1,c1): %s\n"%(pd1c1))

#### P(r1|d1,c1)
print("P(r1|d1,c1)): %s\n"%(pr1d1c1/pd1c1))



#### P(r1,d1,a0)
query = [None, '1', None, '0', '1',  None]
pr1d1a0 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            pr1d1a0 += model.probability([[str(j1), '1', str(j2), '0', '1', str(j3)]])
print("P(r1,d1,a0): %s\n"%(pr1d1a0))

#### P(r1,d1,a1)
query = [None, '1', None, '1', '1',  None]
pr1d1a1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            pr1d1a1 += model.probability([[str(j1), '1', str(j2), '1', '1', str(j3)]])
print("P(r1,d1,a1): %s\n"%(pr1d1a1))

#### P(d1,a0)
query = [None, None, None, '0',  '1',  None]
pd1a0 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                pd1a0 += model.probability([[str(j1), str(j2), str(j3), '0', '1', str(j4)]])
print("P(d1,a0): %s\n"%(pd1a0))

#### P(d1,a1)
query = [None, None, None, '1',  '1',  None]
pd1a1 = 0
for j1 in range(2):
    for j2 in range(2):
        for j3 in range(2):
            for j4 in range(2):
                pd1a1 += model.probability([[str(j1), str(j2), str(j3), '1', '1', str(j4)]])
print("P(d1,a1): %s\n"%(pd1a1))


#### P(r1|d1,a0)
print("P(r1|d1,a0)): %s\n"%(pr1d1a0/pd1a0))

#### P(r1|d1,a1)
print("P(r1|d1,a1)): %s\n"%(pr1d1a1/pd1a1))

# end of intercausal reasoning