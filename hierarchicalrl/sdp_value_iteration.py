"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Todor Davchev, 2017
t.b.davchev@ed.ac.uk
"""

import numpy as np


def value(policy, n_states, transition_probabilities, reward, discount,
          threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v


def optimal_value(option_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    value = np.zeros(len(option_states))

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for idx, state in enumerate(option_states):
            max_v = float("-inf")
            for action in range(n_actions):
                transition_p = transition_probabilities[state, action, :]
                transition_p = [
                    [
                        x for br, x in enumerate(transition_p) if br == opt_state]
                    for opt_state in option_states]
                transition_p = np.asarray(transition_p)
                transition_p = np.reshape(transition_p, transition_p.shape[0])
                max_v = max(max_v, sum(
                    reward + np.dot(transition_p, (discount * value))))
                # max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(value[idx] - max_v)
            if new_diff > diff:
                diff = new_diff
            value[idx] = max_v


        # diff = 0
        # for s in range(n_states):
        #     max_v = float("-inf")
        #     for a in range(n_actions):
        #         tp = transition_probabilities[s, a, :]
        #         max_v = max(max_v, np.dot(tp, reward + discount*v))

        #     new_diff = abs(v[s] - max_v)
        #     if new_diff > diff:
        #         diff = new_diff
        #     v[s] = max_v


    return value


def optimal_value_option(options_states, n_options, options_transition_probabilities,
                         reward_o, discount, threshold=1e-2):
    value_o = np.zeros(121)
    diff_o = float("inf")
    while diff_o > threshold:
        diff_o = 0
        for state in range(121):
            max_vo = float("-inf")
            for option in range(8):
                transition_po = options_transition_probabilities[state, option, :]
                transition_po = np.asarray(transition_po)
                transition_po = np.reshape(transition_po, transition_po.shape[0])
                #  [filter(lambda x: x in c1, sublist)
                #                 for sublist in c2]
                max_vo = max(max_vo,
                    reward_o[option] + np.dot(transition_po, (discount * value_o)))
                # max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff_o = abs(value_o[state] - max_vo)
            if new_diff_o > diff_o:
                diff_o = new_diff_o
            value_o[state] = max_vo

    return value_o

def optimal_option_value(option_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """

    value = np.zeros(len(option_states))

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for idx, state in enumerate(option_states):
            max_v = float("-inf")
            for action in range(n_actions):
                transition_p = transition_probabilities[state, action, :]
                transition_p = [[x for br, x in enumerate(transition_p) if br == state] for state in option_states]
                transition_p = np.asarray(transition_p)
                transition_p = np.reshape(transition_p, transition_p.shape[0])
                #  [filter(lambda x: x in c1, sublist)
                #                 for sublist in c2]
                max_v = max(max_v, sum(
                    reward + np.dot(transition_p[0], (discount * value))))
                # max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(value[idx] - max_v)
            if new_diff > diff:
                diff = new_diff
            value[idx] = max_v

    return value

# def optimal_value(n_states, n_actions, transition_probabilities, reward,
#                   discount, threshold=1e-2):
#     """
#     Find the optimal value function.

#     n_states: Number of states. int.
#     n_actions: Number of actions. int.
#     transition_probabilities: Function taking (state, action, state) to
#         transition probabilities.
#     reward: Vector of rewards for each state.
#     discount: MDP discount factor. float.
#     threshold: Convergence threshold, default 1e-2. float.
#     -> Array of values for each state
#     """

#     v = np.zeros(n_states)

#     diff = float("inf")
#     while diff > threshold:
#         diff = 0
#         for s in range(n_states):
#             max_v = float("-inf")
#             for a in range(n_actions):
#                 tp = transition_probabilities[s, a, :]
#                 # max_v = max(max_v, sum(reward + np.dot(tp, discount*v)))
#                 max_v = max(max_v, np.dot(tp, reward + discount*v))

#             new_diff = abs(v[s] - max_v)
#             if new_diff > diff:
#                 diff = new_diff
#             v[s] = max_v

#     return v


def find_option_policy(options_states, n_states, n_actions, n_options, options_transition_probabilities,
                transition_probabilities, reward_o, reward, discount,
                threshold=1e-2, value=None, stochastic=True):
    q_values = []#np.zeros((len(options_states), n_states, n_actions))
    if value is None:
        option_value = optimal_value_option(options_states, n_options,
                                                 options_transition_probabilities,
                                                 reward_o, discount, threshold)

    if stochastic:
        options_Q = np.zeros((121, n_options))
        for i in range(121):
            for j in range(n_options):
                p = options_transition_probabilities[i, j, :]
                options_Q[i, j] = reward_o[j] + p.dot(discount*option_value)
        options_Q -= options_Q.max(axis=1).reshape((121, 1))  # For numerical stability.
        options_Q = np.exp(options_Q)/np.exp(options_Q).sum(axis=1).reshape((121, 1))
        return options_Q

def find_policy(options_states, n_states, n_actions, n_options, options_transition_probabilities,
                transition_probabilities, reward_o, reward, discount,
                threshold=1e-2, value=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    q_values = []#np.zeros((len(options_states), n_states, n_actions))
    if value is None:
        value = []
        option_value = []
        for option, option_states in enumerate(options_states):
            value.append(
                optimal_value(
                    option_states, n_actions, transition_probabilities[option],
                    reward[option], discount, threshold))

    if stochastic:
        for option, option_states in enumerate(options_states):
            q_values.append(np.zeros((len(option_states), n_actions)))
            # Get Q using equation 9.2 from Ziebart's thesis.
            for idx, i_state in enumerate(option_states):
                for j_action in range(n_actions):
                    transition_p = transition_probabilities[option, i_state, j_action, :]
                    transition_p = [[
                        x for br, x in enumerate(transition_p) if br == opt_state]
                                    for opt_state in option_states]
                    transition_p = np.asarray(transition_p)
                    transition_p = np.reshape(transition_p, transition_p.shape[0])
                    q_values[option][idx, j_action] = sum(reward[option] + transition_p.dot(
                        discount * value[option]))
            q_values[option] -= q_values[option].max(axis=1).reshape((n_states[option], 1))
            # For numerical stability.
            q_values[option] = np.exp(
                q_values[option]) / np.exp(q_values[option]).sum(axis=1).reshape(
                    (n_states[option], 1))

        # Q = np.zeros((n_states, n_actions))
        # for i in range(n_states):
        #     for j in range(n_actions):
        #         p = transition_probabilities[i, j, :]
        #         Q[i, j] = p.dot(reward + discount*v)
        # Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        # Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        # return Q

        return q_values

    # def _policy(s):
    #     return max(range(n_actions),
    #                key=lambda a: sum(transition_probabilities[s, a, k] *
    #                                  (reward[k] + discount * v[k])
    #                                  for k in range(n_states)))
    # policy = np.array([_policy(s) for s in range(n_states)])
    # return policy


if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    assert np.isclose(v, opt_v).all()
