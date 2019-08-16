"""
Implements Semi-MDP maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Todor Davchev, 2017
t.b.davchev@ed.ac.uk
"""

from itertools import product

import numpy as np
import numpy.random as rn

import sdp_value_iteration as value_iteration

def irl(options_states, features_matrix, o_feature_matrix, n_actions, n_options, discount,
        options_transition_probability, transition_probability,
        trajectories, global_trajectories, epochs, learning_rate, int_to_point, options):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    -> Reward vector with shape (N,).
    """

    n_states = [np.asarray(i).shape[0] for i in features_matrix]
    d_states = [np.asarray(i).shape[1] for i in features_matrix]
    on_states = o_feature_matrix.shape[0]
    od_states = o_feature_matrix.shape[1]
    # n_states, d_states = [features_matrix[i].shape for i in xrange(features_matrix)]

    # Initialise weights.
    alpha = [rn.uniform(size=(d_st,)) for d_st in d_states]
    o_alpha = rn.uniform(size=(od_states,))

    # option = 0
    # Calculate the feature expectations \tilde{phi}.

    # change the samples to go from option through option etc ...
    feature_expectations, options_feature_expectations = find_feature_expectations(
        features_matrix, o_feature_matrix, trajectories, global_trajectories, options_states)

    # Gradient descent on alpha.
    for i in range(epochs):
        # print("i: {}".format(i))
        r = np.asarray([np.asarray(features_matrix[opt]).dot(alpha[opt]) for opt in range(n_options)])
        r_o = o_feature_matrix.dot(o_alpha)
        expected_svf, options_expected_svf = find_expected_svf(
            options_states, on_states, n_states, r_o, r,
            n_actions, n_options, discount, options_transition_probability,
            transition_probability, trajectories, global_trajectories)
        #not for 0 only but for all options
        modif_expected_svf = [
            [
                [
                    item for idx, item in enumerate(expected_svf[opt]) if idx == opt_state]
                for opt_state in options_states[opt]]
            for opt in range(n_options)]
        grad = [feature_expectations[opt] - np.asarray(features_matrix[opt]).T.dot(modif_expected_svf[opt]).reshape((n_states[opt],)) for opt in range(n_options)]
        modif_opt_exp_svf = [[value for idx, value in enumerate(options_expected_svf) if int_to_point(idx) == opt["termination"]][0] for opt in options]
        o_grad = options_feature_expectations - o_feature_matrix.T.dot(modif_opt_exp_svf)

        alpha += [learning_rate * grad[opt] for opt in range(n_options)]
        o_alpha += learning_rate * o_grad

    return [np.asarray(features_matrix[opt]).dot(alpha[opt]).reshape((n_states[opt],)) for opt in range(n_options)],\
        o_feature_matrix.dot(o_alpha).reshape((n_options,))

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, o_feature_matrix, trajectories, global_trajectories, options_states):
    """
    Find the feature expectations for the given trajectories. This is the
    average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. NumPy
        array with shape (N, D) where N is the number of states and D is the
        dimensionality of the state.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Feature expectations vector with shape (D,).
    """
    option_feature_expectations = np.zeros(len(o_feature_matrix)) # kolko optioni ima i v koi e bil nai-mnogo
    feature_expectations = [np.zeros(len(feature_matrix[i])) for i in xrange(len(options_states))]
    for br, option_states in enumerate(options_states):
        for trajectory in trajectories[br]:
            # for state, _, _ in trajectory:
            for traj_id in trajectory:
                feature_expectations[br] += feature_matrix[br][
                    [idx for idx, state in enumerate(option_states) if state == traj_id[0]][0]]

        feature_expectations[br] /= trajectories[br].shape[0]

    for global_traj in global_trajectories:
        for option_used in global_traj:
            option_feature_expectations += o_feature_matrix[option_used[1]]

    option_feature_expectations /= global_trajectories.shape[0]

    return feature_expectations, option_feature_expectations

def find_expected_svf(options_states, on_states, n_states, r_o, r, n_actions, n_options, discount,
                      options_transition_probability, transition_probability, trajectories,
                      global_trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states N. int.
    alpha: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> Expected state visitation frequencies vector with shape (N,).
    """

    n_trajectories = trajectories[0].shape[0]
    trajectory_lengths = trajectories[0].shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)

    # policy = [[] for _ in range(len(options_states))]
    policy = value_iteration.find_policy(options_states, n_states, n_actions, n_options,
                                         options_transition_probability,
                                         transition_probability, r_o, r, discount)

    options_policy = value_iteration.find_option_policy(
        options_states, n_states, n_actions, n_options, options_transition_probability,
        transition_probability, r_o, r, discount)

    # option-to-option
    opt_start_state_count = np.zeros(121)
    global_trajectory_length = 0
    length = 0
    for gl_trajectory in global_trajectories:
        count = 0
        opt_start_state_count[gl_trajectory[0][0][0]] += 1
        for trajectory in gl_trajectory:
            count += len(trajectories[0])

        if count > length:
            length = count

    op_start_state = opt_start_state_count/n_trajectories

    opt_expected_svf = np.tile(op_start_state, (length, 1)).T
    for t in range(1, length):
        opt_expected_svf[:, t] = 0
        for i, j, k in product(range(121), range(n_options), range(121)):
            opt_expected_svf[k, t] += (opt_expected_svf[i, t-1] *
                                  options_policy[i, j] * # Stochastic policy
                                  options_transition_probability[i, j, k])

    options_result = opt_expected_svf.sum(axis=1)

    # intra-options
    start_state_count = np.zeros((8, 121))
    p_start_state = []
    for option in range(n_options):
        for trajectory in trajectories[option]:
            start_state_count[option][trajectory[0, 0]] += 1
        p_start_state.append(start_state_count[option]/n_trajectories)
    result = []
    expected_svf = [np.tile(p_start_state[opt], (trajectory_lengths, 1)).T for opt in range(len(options_states))]
    ids = [[
            56, 45, 44, 46, 47, 48,
            33, 34, 35, 36, 37,
            22, 23, 24, 25, 26, 27,
            11, 12, 13, 14, 15,
            0, 1, 2, 3, 4
        ],
        [
            27, 26, 15, 4, 37, 48,
            3, 14, 25, 36, 47,
            2, 13, 24, 35, 46,
            1, 12, 23, 34, 45,
            0, 11, 22, 33, 44, 56
        ],
        [
            27, 28, 17, 39, 17, 6, 39, 50, 61,
            62, 51, 40, 29, 18, 7,
            8, 19, 30, 41, 52, 63, 74,
            9, 20, 31, 42, 53, 64,
            10, 21, 32, 43, 54, 65
        ],
        [
            74, 63, 62, 61, 64, 65,
            50, 51, 52, 53, 54,
            39, 40, 41, 42, 43,
            32, 31, 30, 29, 28, 27,
            17, 18, 19, 20, 21, 6, 7, 8, 9, 10
        ],
        [
            74, 85, 84, 83, 86, 87,
            94, 95, 96, 97, 98,
            109, 108, 107, 106, 105, 104,
            116, 117, 118, 119, 120
        ],
        [
            104, 105, 116, 94, 83,
            84, 95, 106, 117,
            118, 107, 96, 85, 74,
            86, 97, 108, 119,
            87, 98, 109, 120
        ],
        [
            104, 103, 114, 92, 81, 70,
            69, 80, 91, 102, 113,
            68, 79, 90, 101, 112,
            67, 56, 78, 89, 100, 111,
            66, 77, 88, 99, 110
        ],
        [
            56, 67, 66, 68, 69, 70,
            77, 78, 79, 80, 81,
            88, 89, 90, 91, 92,
            99, 100, 101, 102, 103, 104,
            110, 111, 112, 113, 114
        ]]
    for o in range(len(ids)):
        for t in range(1, trajectory_lengths):
            expected_svf[o][:, t] = 0
            for i, j, k in product(ids[o], range(n_actions),ids[o]):
                if i in options_states[o]:
                    idme = [idx for idx, state in enumerate(options_states[o]) if state == i][0]
                    # Stochastic policy
                    expected_svf[o][k, t] += (expected_svf[o][i, t-1] * policy[o][idme, j] *
                                              transition_probability[o][i, j, k])
                else:
                    expected_svf[o][k, t] = 0

        result.append(expected_svf[o].sum(axis=1))

    return result, options_result

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def expected_value_difference(n_states, n_actions, transition_probability,
    reward, discount, p_start_state, optimal_value, true_reward):
    """
    Calculate the expected value difference, which is a proxy to how good a
    recovered reward function is.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    reward: Reward vector mapping state int to reward. Shape (N,).
    discount: Discount factor. float.
    p_start_state: Probability vector with the ith component as the probability
        that the ith state is the start state. Shape (N,).
    optimal_value: Value vector for the ground reward with optimal policy.
        The ith component is the value of the ith state. Shape (N,).
    true_reward: True reward vector. Shape (N,).
    -> Expected value difference. float.
    """

    policy = value_iteration.find_policy(n_states, n_actions,
        transition_probability, reward, discount)
    value = value_iteration.value(policy.argmax(axis=1), n_states,
        transition_probability, true_reward, discount)

    evd = optimal_value.dot(p_start_state) - value.dot(p_start_state)
    return evd
