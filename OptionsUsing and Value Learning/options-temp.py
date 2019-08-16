import numpy as np
import random


Sx = 5
Sy = 5
S = Sx*Sy
P = 5 # there is a state for being in the taxi
G = 4
R = 0
maxR = -999999
maxRo = -999999
pickUps = [0, Sx-1, S-Sx, S-2]
O = 7
A = 6
T = 100000
k_step = 0
stepNo = 0
stepNo_o = 0
avg_reward = np.zeros([Sx, Sy, P, G])
reward = np.zeros([P, G])
avg = np.zeros([T,1])
time_course = np.zeros([T, 3])
Q = 0.1*np.random.rand(S, O, P, G)
opt_one_policy = 0.1*np.random.rand(S, A, P, G)
V = np.max(Q, axis=1)
V_o = np.max(opt_one_policy, axis=1)
eta = 0.1
gamma = 0.9
epsilon = 0.3
reward_course = np.zeros([T, 1])
reward_mean = np.zeros([T, 1])
shouldBreak = False

stop = True
stepsToGoal = np.zeros([T, 1])
maxV = -9999
maxV_o = -9999
for t in xrange(T):
    plocation = 20
    pID = [i for i, x in enumerate(pickUps) if x == plocation][0]
    Goal = 24
    p0 = plocation + 1
    gID = [i for i, x in enumerate(pickUps) if x == Goal-1][0]
    s0 = np.random.choice(S)
    state = [s0, pID, gID] #[{1..25} {1..5} {1..4}]
    for u in xrange(S**4):
        if (stepNo > 30):
            stepNo = 0
            stepNo_o = 0
            break

        r = 0
        if stop: #V[s1, pID, gID] = np.max(Q[s1, :, pID, gID])
            # if o0 == 6: # if previous time had the option and now time's gone
            #     V[s1, pID, gID] = np.max(Q[s1, :, pID, gID])
            [V[s0, pID, gID], o0] = [np.max(Q[s0, :, pID, gID]), np.argmax(Q[s0, :, pID, gID])]

            if (np.random.rand(1) < epsilon):
                o0 = np.random.choice(O)

            if o0 == 6:
                r_o = 0
                k_step = 1
                stop = False
                [V_o[s0, pID, gID], a0] = [np.max(opt_one_policy[s0, :, pID, gID]),
                                           np.argmax(opt_one_policy[s0, :, pID, gID])]
        else:
            k_step += 1
            [V_o[s0, pID, gID], a0] = [
                np.max(opt_one_policy[s0, :, pID, gID]), np.argmax(opt_one_policy[s0, :, pID, gID])]

        if o0 == 4:
            if pID != 4:
                if s0 == pickUps[pID]:
                    r = 1
                    pID = 4
                    stepNo = 0
                else:
                    r = -1
            else:
                r = -1

        if o0 == 5:
            if (s0 == pickUps[gID]) and pID==4:
                stepsToGoal[t] = stepNo
                r = 10/float(stepNo)
                if maxR < r:
                    maxR = r

                stepNo = 0
            else:
                r = -1

        if o0 == 0:
            s1 = s0 - Sx
            if s1 < 0:
                s1 = s1 + Sx
                r = -1

        if o0 == 1:
            s1 = s0 + Sx
            if s1 > 24:
                s1 = s1 - Sx
                r = -1

        if o0 == 2:
            s1=s0-1
            if s1==-1 or s1==4 or s1==9 or s1==14 or s1==19:
                s1=s1+1
                r = -1

            if s1==1 or s1==6 or s1==20 or s1==15 or s1==17 or s1==22:
                s1 = s1+1
                r = -1

        if o0 == 3:
            s1 = s0 + 1
            if s1 == 5 or s1 == 10 or s1 == 15 or s1 == 20 or s1==25:
                s1 = s1 - 1
                r = -1

            if s1 == 2 or s1 == 7 or s1 == 21 or s1 == 16 or s1 == 18 or s1 == 23:
                s1 = s1 - 1
                r = -1

        if o0 == 6:
            if pID != 4:
                s1 = s0
                stop = True
            else:
                stepNo_o += 1
                if a0 == 4:
                    r_o = -1

                if a0 == 5:
                    if (s0 == 16): # 19 is a random state I would like my option to get to
                        # stepsToGoal[t] = stepNo
                        r_o = 1
                        if maxRo < r_o:
                            maxRo = r_o

                        stop = True
                        stepNo_o = 0
                    # elif (s0 == pickUps[gID]):
                    #     # stepsToGoal[t] = stepNo
                    #     r_o = 10/float(stepNo)
                    #     if maxR_o < r_o:
                    #         maxR_o = r_o

                    #     stop = True
                    #     stepNo = 0
                    else:
                        r_o = -1

                if a0 == 0:
                    s1 = s0 - Sx
                    if s1 < 0:
                        s1 = s1 + Sx
                        r_o = -1

                if a0 == 1:
                    s1 = s0 + Sx
                    if s1 > 24:
                        s1 = s1 - Sx
                        r_o = -1

                if a0 == 2:
                    s1=s0-1
                    if s1==-1 or s1==4 or s1==9 or s1==14 or s1==19:
                        s1=s1+1
                        r_o = -1

                    if s1==1 or s1==6 or s1==20 or s1==15 or s1==17 or s1==22:
                        s1 = s1+1
                        r_o = -1

                if a0 == 3:
                    s1 = s0 + 1
                    if s1 == 5 or s1 == 10 or s1 == 15 or s1 == 20 or s1==25:
                        s1 = s1 - 1
                        r_o = -1

                    if s1 == 2 or s1 == 7 or s1 == 21 or s1 == 16 or s1 == 18 or s1 == 23:
                        s1 = s1 - 1
                        r_o = -1

                if a0 == 4:
                    s1 = s0

                if a0 == 5:
                    s1 = s0

                if (s0 == pickUps[gID]) and (a0 == 5):
                    shouldBreak = True

        if o0 == 4:
            s1 = s0

        if o0 == 5:
            s1 = s0

        if stop:
            # print "tuk sam"
            if o0 == 6:
                r = r_o
                # print r
            # learning step
            if t > 1000:
                R += r

            # print r
            FullR = R + r
            reward_course[t] = r
            reward_mean[t] = R/float(t+1)

            V[s1, pID, gID] = np.max(Q[s1, :, pID, gID])

            if maxV < V[s1, pID, gID]:
                maxV = V[s1, pID, gID]

            time_course[t, 0] = V[s1, pID, gID]
            time_course[t, 1] = eta*(r+gamma*V[s1, pID, gID])
            time_course[t, 2] = (1-eta)*Q[s0, o0, pID, gID]

            Q[s0, o0, pID, gID] = (1-eta)*Q[s0, o0, pID, gID] + eta*(r + gamma*V[s1, pID, gID])
        else:
            V_o[s1, pID, gID] = np.max(opt_one_policy[s1, :, pID, gID])
            opt_one_policy[s0, a0, pID, gID] = (1-eta)*opt_one_policy[s0, a0, pID, gID] +\
                eta*(r + gamma*V_o[s1, pID, gID])
            if maxV_o < V_o[s1, pID, gID]:
                maxV_o = V_o[s1, pID, gID]

        if pID == 4:
            stepNo += 1

        if (s0 == pickUps[gID]) and (o0 == 5):
            stepNo = 0
            break


        s0 = s1


    avg[t] = np.mean(np.mean(np.mean(np.mean(Q))))


meanR = R/float(T-1000)
fullMR = FullR/float(T)
print meanR
print fullMR
print maxV
print "Policy"
policy = [np.max(Q[i, :, pID, gID]) for i in xrange(S)]
policy_actions = [np.argmax(Q[i, :, pID, gID]) for i in xrange(S)]
policy_inoption = [np.argmax(opt_one_policy[i, :, pID, gID]) for i in xrange(S)]
print len(policy)
policy_actions = np.reshape(policy_actions, [5, 5])
policy_inoption = np.reshape(policy_inoption, [5, 5])


for i in xrange(5):
    for j in xrange(5):
        print "{0} ".format(policy_actions[i, j]),


    print " "

print "In option actions"
for i in xrange(5):
    for j in xrange(5):
        print "{0} ".format(policy_inoption[i, j]),

    print " "
