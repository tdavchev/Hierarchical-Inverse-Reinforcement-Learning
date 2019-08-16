import numpy as np
import random


Sx = 7
Sy = 7
S = Sx*Sy
P = 5 # there is a state for being in the taxi
G = 4
R = 0
O = 2
maxR = -999999
hallways = [44, 27]
rooms = [[
    8, 9, 10, 11, 12,
    15, 16 ,17, 18, 19,
    22, 23, 24, 25, 26, 27,
    36, 37, 38, 39, 40,
    44
]]

walls = [[
    0, 1, 2, 3, 4, 5, 6,
    7, 14, 21, 28, 35, 42,
    43, 45, 46, 47, 48,
    13, 20, 34, 41, 48,
    51
]]

# init_states = [0, 0, 1, 1, 2, 2, 3, 3]
pickUps = [44, 27]
A = 6
T = 3000
stepNo = 0
avg_reward = np.zeros([Sx, Sy, P, G])
reward = np.zeros([P, G])
avg = np.zeros([T,1])
time_course = np.zeros([T, 3])
Q = 0.1*np.random.rand(S, O, A, P, G)
for i in xrange(49):
    for o in xrange(O):
        for a in xrange(A):
            for p in xrange(P):
                for g in xrange(G):
                    if i not in rooms[0]:
                        Q[i, o, a, p, g] = 0

V = [np.max(Q[:, o, :], axis=1) for o in xrange(O)]
eta = 0.1
gamma = 0.9
epsilon = 0.1
reward_course = np.zeros([T, 1])
reward_mean = np.zeros([T, 1])

option = 0

stepsToGoal = np.zeros([T, 1])
maxV = -9999
for t in xrange(T):
    plocation = pickUps[1 - option]
    pID = [i for i, x in enumerate(pickUps) if x == plocation][0]
    Goal = pickUps[option]
    p0 = plocation
    gID = [i for i, x in enumerate(pickUps) if x == Goal][0]
    s0 = np.random.choice([state for state in xrange(S) if state not in walls[0]])
    state = [s0, pID, gID] #[{1..25} {1..5} {1..4}]
    for u in xrange(S**2):
        if (stepNo > 30):
            stepNo = 0
            break

        r = 0
        [V[option][s0, pID, gID], a0] = [np.max(Q[s0, option, :, pID, gID]), np.argmax(Q[s0, option, :, pID, gID])]
        if (np.random.rand(1) < epsilon):
            a0 = np.random.choice(A)


        if a0 == 4:
            if pID != 4:
                if s0 == pickUps[pID]:
                    r = 1
                    pID = 4
                    stepNo = 0
                else:
                    r = -1
            else:
                r = -1


        if a0 == 5:
            if (s0 == pickUps[gID]) and pID==4:
                stepsToGoal[t] = stepNo
                r = 10/float(stepNo)
                if maxR < r:
                    maxR = r

                stepNo = 0
            else:
                r = -1


        if a0 == 0:
            s1 = s0 - Sx
            if s1 in walls[0]:
                s1 = s1 + Sx
                r = -1


        if a0 == 1:
            s1 = s0 + Sx
            if s1 in walls[0]:
                s1 = s1 - Sx
                r = -1


        if a0 == 2:
            s1 = s0 - 1
            if s1 in walls[0]:
                s1 = s1 + 1
                r = -1


        if a0 == 3:
            s1 = s0 + 1
            if s1 in walls[0]:
                s1 = s1 - 1
                r = -1

        if a0 == 4:
            s1 = s0


        if a0 == 5:
            s1 = s0


        # learning step
        if t > 1000:
            R += r


        # print r
        FullR = R + r
        reward_course[t] = r
        reward_mean[t] = R/float(t+1)


        V[option][s1, pID, gID] = np.max(Q[s1, option, :, pID, gID])


        if maxV < V[option][s1, pID, gID]:
            maxV = V[option][s1, pID, gID]


        time_course[t, 0] = V[option][s1, pID, gID]
        time_course[t, 1] = eta*(r+gamma*V[option][s1, pID, gID])
        time_course[t, 2] = (1-eta)*Q[s0, option, a0, pID, gID]
        Q[s0, option, a0, pID, gID] = (1-eta)*Q[s0, option, a0, pID, gID] + \
            eta*(r + gamma*V[option][s1, pID, gID])
        if pID == 4:
            stepNo += 1


        if (s0 == pickUps[gID]) and (a0 == 5):
            stepNo = 0
            break


        s0 = s1


    avg[t] = np.mean(np.mean(np.mean(np.mean(Q[:, option, :]))))


meanR = R/float(T-1000)
fullMR = FullR/float(T)
print meanR
print fullMR
print maxV
policy = [np.max(Q[i, option, :, pID, gID]) for i in xrange(S)]
policy_actions = [np.argmax(Q[i, option, :, pID, gID]) for i in xrange(S)]
print len(policy)
policy_actions = np.reshape(policy_actions, [7, 7])


for i in xrange(7):
    for j in xrange(7):
        print "{0} ".format(policy_actions[i, j]),

    print " "