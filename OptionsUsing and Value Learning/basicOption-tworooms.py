import numpy as np
import random


Sx = 13
Sy = 8
S = Sx*Sy
P = 5 # there is a state for being in the taxi
G = 4
R = 0
O = 4
maxR = -999999
hallways = [80, 45, 100]
rooms = [[
    14, 15, 16, 17, 18,
    27, 28, 29, 30, 31,
    40, 41, 42, 43, 44, 45,
    53, 54, 55, 56, 57,
    66, 67, 68, 69, 70,
    80
],
[
    20, 21, 22, 23, 24,
    33, 34, 35, 36, 37,
    45, 46, 47, 48, 49, 50,
    59, 60, 61, 62, 63,
    72, 73, 74, 75, 76,
    85, 86, 87, 88, 89,
    100
]]

walls = [[
    0, 1, 2, 3, 4, 5, 6,
    13, 26, 39, 52, 65,
    78, 79, 81, 82, 83,
    84, 71, 58, 32, 19,
    93
],
[
    7, 8, 9, 10, 11, 12,
    25, 38, 51, 64, 77, 90, 103,
    98, 99, 101, 102, 103,
    97, 84, 71, 58, 32, 19, 6,
    113
]]

# init_states = [0, 0, 1, 1, 2, 2, 3, 3]
pickUps = [80, 45, 45, 100]
A = 6
T = 30000
stepNo = 0
avg_reward = np.zeros([Sx, Sy, P, G])
reward = np.zeros([P, G])
avg = np.zeros([T,1])
time_course = np.zeros([T, 3])
options_used = []
option = 0
options_used.append(option)
option_goal = [45, 80, 100, 45]
endGoal = option_goal[2]
room_no = [0, 0, 1, 1]
Q = 0.1*np.random.rand(S, O, A, P, G)
for i in xrange(S):
    for o in xrange(O):
        for a in xrange(A):
            for p in xrange(P):
                for g in xrange(G):
                    if i not in rooms[room_no[o]]:
                        Q[i, o, a, p, g] = 0

V = [np.max(Q[:, o, :], axis=2) for o in xrange(O)]
eta = 0.1
gamma = 0.9
epsilon = 0.1
reward_course = np.zeros([T, 1])
reward_mean = np.zeros([T, 1])

stepsToGoal = np.zeros([T, 1])
maxV = -9999
switched = False
for t in xrange(T):
    Goal = option_goal[option]
    gID = [i for i, x in enumerate(pickUps) if x == Goal][0]
    if not switched:
        plocation = pickUps[option]
        p0 = plocation
        pID = [i for i, x in enumerate(pickUps) if x == plocation][0]
        s0 = np.random.choice([state for state in xrange(S) if state in rooms[room_no[option]]])

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
                if stepNo > 0:
                    r = 10
                else:
                    r = 10
                if maxR < r:
                    maxR = r

                stepNo = 0
            else:
                r = -1


        if a0 == 0:
            s1 = s0 - Sx
            if s1 not in rooms[room_no[option]]:
                s1 = s1 + Sx
                r = -1


        if a0 == 1:
            s1 = s0 + Sx
            if s1 not in rooms[room_no[option]]:
                s1 = s1 - Sx
                r = -1


        if a0 == 2:
            s1 = s0 - 1
            if s1 not in rooms[room_no[option]]:
                s1 = s1 + 1
                r = -1


        if a0 == 3:
            s1 = s0 + 1
            if s1 not in rooms[room_no[option]]:
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

        if (s0 == endGoal) and (a0 == 5) and (pID == 4):
            stepNo = 0
            option = 0
            switched = False
            # print "Final Goal achieved!!"
            break

        if (s0 == pickUps[gID]) and (a0 == 5) and (pID == 4) and s0 != endGoal:
            stepNo = 0
            # print "---> ",
            # print option
            option = 2
            switched = True
            options_used.append(option)
            break

        s0 = s1


    avg[t] = np.mean(np.mean(np.mean(np.mean(Q[:, option, :]))))


meanR = R/float(T-1000)
fullMR = FullR/float(T)
print meanR
print fullMR
print maxV
policy = [np.max(Q[i, option, :, pID, gID]) for i in xrange(S)]
policy_actions_0 = [np.argmax(Q[i, 0, :, pID, gID]) for i in xrange(S)]
policy_actions_2 = [np.argmax(Q[i, 2, :, pID, gID]) for i in xrange(S)]
print len(policy)
policy_actions_0 = np.reshape(policy_actions_0, [8, 13])
policy_actions_2 = np.reshape(policy_actions_2, [8, 13])
policy = np.reshape(policy, [8, 13])

for j in xrange(8):
    for i in xrange(13):
        print "{0} ".format(policy_actions_0[j, i]),

    print " "
print "-------------------------------------"
for j in xrange(8):
    for i in xrange(13):
        print "{0} ".format(policy_actions_2[j, i]),

    print " "
# for j in xrange(8):
#     for i in xrange(13):
#         print "{0} ".format(int(policy[j, i])),

#     print " "