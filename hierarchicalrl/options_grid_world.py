"""
Implements the options gridworld MDP.

Todor Davchev, 2017
t.b.davchev@ed.ac.uk
"""

import numpy as np
import numpy.random as rn


class Large_Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, walls, options, rooms, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """
        self.count = 0
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.options = options
        self.rooms = rooms
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount
        self.walls = walls
        self.init_states = [(5, 2), (1, 5), (8, 6), (5, 9)]
        self.term_states = [(5, 2), (1, 5), (8, 6), (5, 9)] 
        self.n_options = len(self.options)
        self.k_length = 10
        self.gamma = 0.9
        self.count = 0
        # Preconstruct the transition probability array.
        # self.transition_probability = np.array(
        #     [[[self._transition_probability(i, j, k)
        #        for k in range(self.n_states)]
        #       for j in range(self.n_actions)]
        #      for i in range(self.n_states)])

        # mislq che e greshno
        # self.reward_o = np.array(
        #     [self._reward_o(option)
        #      for option in self.options[:2]]
        # )

        # Preconstruct the transition probability array.
        self.improved_transition_probability = np.array(
            [[[[self._improved_transition_probability(o, i, j, k)
                for k in range(self.n_states)]
               for j in range(self.n_actions)]
              for i in range(self.n_states)]
             for o in range(self.n_options)])
            #  for o in range(self.n_options)])

        # Preconstruct the transition probability array.
        # after done, initial states are all within a room
        # should factor in sudden change of option ? - yes
        # self.options_transition_probability = np.array(
        #     [[[self._options_transition_probability(i, j, k)
        #        for k in self.init_states]
        #       for j in range(self.n_options)]
        #      for i in self.init_states])
        states_per_option = [[state for state in self.rooms[opt["room"]]] for opt in self.options]
        self.options_transition_probability = np.zeros((121, 8, 121))
        for opt in self.options:
            print opt["id"]
            if opt["id"] == 4:
                print "pochna se"
            za = [
                [
                    self._options_transition_probability(i, opt["id"], k)
                    for k in states_per_option[opt["id"]]
                ]
                for i in states_per_option[opt["id"]]
            ]
            print "compiuted transition prob"
            for br, state in enumerate(states_per_option[opt["id"]]):
                for br_2, state_k in enumerate(states_per_option[opt["id"]]):
                    self.options_transition_probability[state][opt["id"]][state_k] = za[br][br_2]

            print "izlizam ot option.."
        print "Done."

        # self.options_transition_probability = np.array(
        #     [[[self._options_transition_probability(i, o, k)
        #        for k in states_per_option]#[term for term in self.term_states]]
        #       for o in range(self.n_options)]
        #      for i in states_per_option])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    # def _reward_o(self, option):
    #     option_states = self.rooms[option["room"]]
    #     reward_o = [0 for _ in option_states]
    #     for idx, state in enumerate(option_states):
    #         k_step = 1
    #         reward_o[idx] = self.reward(state)
    #         sx, sy = self.int_to_point(state)
    #         # current_state = (sx, sy)
    #         while k_step < self.k_length and (sx, sy) != option["termination"]:
    #             action = self.actions[self.intra_option_optimal_policy(self.point_to_int((sx, sy)), option["id"])]

    #             if ((((sx + action[0],
    #                    sy + action[1]) == (5, 2) and option["id"] == 0)
    #                  or (sx + action[0],
    #                      sy + action[1]) == (1, 5) and option["id"] == 1) or
    #                     (0 <= sx + action[0] < 5 and#self.grid_size and
    #                      0 <= sy + action[1] < 5)):#self.grid_size):
    #                 sx = sx + action[0]
    #                 sy = sy + action[1]

    #             reward_o[idx] += self.reward(self.point_to_int((sx, sy))) * np.power(self.gamma, k_step)
    #             k_step += 1
    #     return reward_o

    def feature_vector(self, i, vec_size, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(vec_size)
        f[i] = 1
        return f

    def o_feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """
        features = []
        for n in range(self.n_options):
            f = self.feature_vector(n, self.n_options, feature_map)
            features.append(f)
        return np.array(features)

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """
# [
#             np.zeros(len(self.rooms[self.options[o]["room"]]))
#             for o in range(self.options)]
        features = [
            [np.zeros(len(self.rooms[self.options[o]["room"]]))
             for __ in range(len(self.rooms[self.options[o]["room"]]))]
            for o in range(len(self.options))]
        # features = np.reshape(features, [len(self.options), self.n_states, self.n_states])
        for o in range(len(self.options)):
            for n in range(len(features[o])):
                # if self.int_to_point(n) not in self.walls: # redundant
                    # idx = [br for br, room in enumerate(self.rooms) if n in room][0]
                    # f = self.feature_vector(n, feature_map)
                    # features[idx][n] = f
                f = self.feature_vector(n, len(self.rooms[self.options[o]["room"]]), feature_map)
                features[o][n] = f
        return np.array(features)

    def opt_to_point(self, i):
        """
        Convert an option int into the corresponding coordinate.

        i: option int.
        -> (x, y) int tuple.
        """

        return self.options[i]["init_set"]

    def point_to_opt(self, p):
        """
        Convert a coordinate into the corresponding state options list.

        p: (x, y) tuple.
        -> State int.
        """

        return [x for x in self.options if x["init_set"] == p]

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1] * self.grid_size

    def isa_wall(self, i):
        """
        Get whether a point is a wall or not. Returns True if wall.

        i: (x, y) int tuple.
        -> bool.
        """

        return i in self.walls

    def neighbouring_option_states(self, i, k):
        """
        Get whether two options neighbour each other. Also returns true if they
        are the same options.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return len([x for x in self.options if x["termination"] == i and x["init_set"] == k]) > 0

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def insame_room(self, i, k):
        """
        Get whether two points are in the same room. Also returns true if they
        are the same point.

        i: int.
        k: int.
        -> [room id].
        """

        item_one = np.asarray([br for br, x in enumerate([i in room for room in self.rooms]) if x])
        item_two = [br for br, x in enumerate([k in room for room in self.rooms]) if x]
        mask = np.in1d(item_one, item_two)

        return item_one[mask]

        # return [item for item in np.in1d(item_one, item_two) if item]

    #  def _options_transition_probability(self, o, i, j, k):
    #     """
    #     Get the probability of transitioning from state i to state k given
    #     action j.

    #     maybe start with option_state, option, option_state
    #     if possible to get there, if the option state is the actual goal
    #     assign 1 - wind, otherwise it should be 50% ?

    #     i: Option State int.
    #     j: Action int.
    #     k: State int.
    #     -> p(s_k | s_i, a_j)
    #     """
    #     options_i = self.point_to_opt(i)
    #     # option_id = [br for br, x in enumerate(self.init_states) if x == i][0]
    #     option_action = self.options[j]
    #     # option_kd = [br for br, x in enumerate(self.init_states) if x == k][0]
    #     options_k = self.point_to_opt(k)

    #     if i != option_action["init_set"]:
    #         if i == k:
    #             return 1.0
    #         else:
    #             return 0.0

    #     if i == k:
    #         return 0.0

    #     if i == option_action["init_set"]:
    #         if k == option_action["termination"]:
    #             return 1 - self.wind

    #         else:
    #             s = [x for x in options_i if x["termination"]
    #                  != option_action["termination"]]
    #             for option in s:
    #                 if option["termination"] == k:
    #                     return self.wind / len(s)

    #             return 0.0

    def _options_transition_probability(self, i, o, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        maybe start with option_state, option, option_state
        if possible to get there, if the option state is the actual goal
        assign 1 - wind, otherwise it should be 50% ?

        i: Option State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xk, yk = self.int_to_point(k)

        if (xk, yk) != self.options[o]["termination"]:
            return 0

        room_no = np.asarray(self.insame_room(i, self.point_to_int((xk, yk))))

        if self.options[o]["room"] not in room_no:
            return 0.0

        k_step = 0
        # current_state = (sx, sy)
        while (xi, yi) != self.options[o]["termination"]:
            action = self.actions[self.intra_option_optimal_policy(
                self.point_to_int((xi, yi)), self.options[o]["id"])]

            if (((xi + action[0],
                  yi + action[1]) == self.options[o]["termination"]) or (
                      self.options[o]["min"][0] < xi + action[0] < self.options[o]["max"][0] and
                      self.options[o]["min"][1] < yi + action[1] < self.options[o]["max"][1])
               ):
                xi = xi + action[0]
                yi = yi + action[1]

            k_step += 1
            if k_step > 8:
                print "wtf"

        return np.power(self.gamma, k_step)

    def _improved_transition_probability(self, o, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        room_no = np.asarray(self.insame_room(i, k))

        if len(room_no) < 1:
            return 0.0

        if self.options[o]["room"] not in room_no:
            return 0.0

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        if self.isa_wall((xi, yi)):
            return 0.0

        if self.isa_wall((xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind / self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind / self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size - 1, self.grid_size - 1),
                        (0, self.grid_size - 1), (self.grid_size - 1, 0),
                        (4, 0), (6, 0), (6, 5), (4, 4), (0, 4), (0, 6), (4, 6),
                        (10, 5), (6, 7), (10, 7), (4, 10), (6, 10)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not ((0 <= xi + xj < self.grid_size and
                     0 <= yi + yj < self.grid_size) and
                    not self.isa_wall((xi + xj, yi + yj))):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2 * self.wind / self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2 * self.wind / self.n_actions
        elif (xi, yi) in {self.int_to_point(27), self.int_to_point(56),
                          self.int_to_point(74), self.int_to_point(104)}:
            if not ((0 <= xi + xj < self.grid_size and
                     0 <= yi + yj < self.grid_size) and
                    not self.isa_wall((xi + xj, yi + yj))):

                if (xi, yi) in self.init_states:
                    return 1 - self.wind/self.n_actions
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2 * self.wind / self.n_actions
            else:
                if (xi, yi) in self.init_states:
                    should_go = np.asarray(self.insame_room(
                        self.point_to_int((xi, yi)),
                        self.point_to_int((xi + xj, yi + yj))))
                    if len(should_go) > 0:
                        if should_go[0] == o:
                            return self.wind - self.wind / self.n_actions

                    return 1 - self.wind/self.n_actions

                # We can blow off the grid in either direction only by wind.
                return 2 * self.wind / self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size - 1} and
                    yi not in {0, self.grid_size - 1} and
                    (xi, yi) not in {
                        self.int_to_point(15), self.int_to_point(
                            37), self.int_to_point(17),
                        self.int_to_point(39), self.int_to_point(
                            50), self.int_to_point(62),
                        self.int_to_point(64), self.int_to_point(
                            86), self.int_to_point(84),
                        self.int_to_point(94), self.int_to_point(
                            92), self.int_to_point(81),
                        self.int_to_point(69), self.int_to_point(
                            68), self.int_to_point(46),
                        self.int_to_point(47)
            }
            ):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size and
                    not self.isa_wall((xi + xj, yi + yj))):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind / self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind / self.n_actions

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        if self.isa_wall((xi, yi)):
            return 0.0

        if self.isa_wall((xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind / self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind / self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size - 1, self.grid_size - 1),
                        (0, self.grid_size - 1), (self.grid_size - 1, 0),
                        (4, 0), (6, 0), (6, 5), (4, 4), (0, 4), (0, 6), (4, 6),
                        (10, 5), (6, 7), (10, 7), (4, 10), (6, 10)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not ((0 <= xi + xj < self.grid_size and
                     0 <= yi + yj < self.grid_size) and
                    not self.isa_wall((xi + xj, yi + yj))):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2 * self.wind / self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2 * self.wind / self.n_actions
        elif (xi, yi) in {self.int_to_point(27), self.int_to_point(56),
                          self.int_to_point(74), self.int_to_point(104)}:
            if not ((0 <= xi + xj < self.grid_size and
                     0 <= yi + yj < self.grid_size) and
                    not self.isa_wall((xi + xj, yi + yj))):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2 * self.wind / self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2 * self.wind / self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size - 1} and
                    yi not in {0, self.grid_size - 1} and
                    (xi, yi) not in {
                        self.int_to_point(15), self.int_to_point(
                            37), self.int_to_point(17),
                        self.int_to_point(39), self.int_to_point(
                            50), self.int_to_point(62),
                        self.int_to_point(64), self.int_to_point(
                            86), self.int_to_point(84),
                        self.int_to_point(94), self.int_to_point(
                            92), self.int_to_point(81),
                        self.int_to_point(69), self.int_to_point(
                            68), self.int_to_point(46),
                        self.int_to_point(47)
            }
            ):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size and
                    not self.isa_wall((xi + xj, yi + yj))):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind / self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind / self.n_actions

    # def reward(self, state_int, option):
    #     """
    #     Reward for being in state state_int.

    #     state_int: State integer. int.
    #     -> Reward.
    #     """
    #     if option == 0 and state_int == 27:#self.n_states - 1:  # self.point_to_int((8, 6)):
    #         return 1
    #     elif option == 1 and state_int == 56:
    #         return 1
    #     return 0

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """
        if state_int == 74:
            return 1

        return 0

    def opt_reward(self, opt_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """
        if opt_int == 2:
            return 1

        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                                  trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory]
                   for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def option_option_optimal_policy(self, option):
        if option == 0:
            return 2
        if option == 1:
            return 7
        if option == 2:
            return 2
        if option == 3:
            return 2
        if option == 4:
            return 5
        if option == 5:
            return 5
        if option == 6:
            return 7
        if option == 7:
            return 5

    def intra_option_optimal_policy(self, state_int, option):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        Actions: {"right": 0, "down": 1, "left": 2, "up": 3}
        -> Action int.
        """
        sx, sy = self.int_to_point(state_int)

        if option == 0:
            if (sx, sy) in [(4, 0), (4, 1), (3, 1)]:
                return 1
            if (sx, sy) in [(3, 4), (4, 4), (4, 3), (1, 5)]:
                return 3
            if (sx, sy) in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
                            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
                            (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4)]:
                return 0

        elif option == 1:
            if (sx, sy) in [(0, 3), (0, 4)]:
                return 0
            if (sx, sy) in [(5, 2), (4, 1), (3, 3), (4, 3), (2, 4), (3, 4), (4, 4)]:
                return 2
            if (sx, sy) in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                            (0, 1), (1, 1), (2, 1), (3, 1),
                            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
                            (1, 3), (2, 3), (1, 4), (1, 5)]:
                return 1

        elif option == 2:
            if (sx, sy) in [(5, 2), (6, 2), (7, 2), (7, 3), (7, 4), (7, 5), (6, 5)]:
                return 0
            if (sx, sy) in [(9, 2), (10, 2), (10, 3), (10, 4), (10, 5), (9, 4), (9, 5)]:
                return 2
            if (sx, sy) in [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
                            (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (8, 2),
                            (8, 3), (8, 4), (8, 5), (8, 6), (6, 3), (6, 4), (9, 3)]:
                return 1

        elif option == 3:
            if (sx, sy) in [(6, 5), (6, 4), (6, 3),
                            (7, 5), (7, 4), (7, 3),
                            (8, 5), (8, 4), (8, 3), (8, 6),
                            (9, 5), (9, 4), (9, 3),
                            (10, 5), (10, 4), (10, 3)]:
                return 3
            if (sx, sy) in [(5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2)]:
                return 2
            if (sx, sy) in [(6, 0), (7, 0), (8, 0), (9, 0), (10, 0),
                            (6, 1), (7, 1), (8, 1), (9, 1), (10, 1)]:
                return 1

        elif option == 4:
            if (sx, sy) in [(6, 7), (7, 7), (8, 7), (9, 7), (10, 7),
                            (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (8, 6)]:
                return 1
            if (sx, sy) in [(5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)]:
                return 2
            if (sx, sy) in [(6, 10), (7, 10), (8, 10), (9, 10), (10, 10)]:
                return 3

        elif option == 5:
            if (sx, sy) in [(6, 7), (7, 7),
                            (6, 8), (7, 8), (6, 9), (5, 9), (7, 9), (6, 10), (7, 10)]:
                return 0
            if (sx, sy) in [(9, 9), (10, 9), (9, 10), (10, 10), (9, 8), (10, 8), (9, 7), (10, 7)]:
                return 2
            if (sx, sy) in [(8, 10), (8, 9), (8, 8), (8, 7), (8, 6)]:
                return 3

        elif option == 6:
            if (sx, sy) in [(0, 6), (0, 7), (0, 8), (0, 9), (0, 10)]:
                return 0
            if (sx, sy) in [(2, 6), (3, 6), (4, 6),
                            (2, 7), (3, 7), (4, 7),
                            (2, 8), (3, 8), (4, 8),
                            (2, 9), (3, 9), (4, 9),
                            (2, 10), (3, 10), (4, 10), (5, 9)]:
                return 2
            if (sx, sy) in [(1, 10), (1, 9), (1, 8), (1, 7), (1, 6), (1, 5)]:
                return 3

        elif option == 7:
            if (sx, sy) in [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9)]:
                return 0
            if (sx, sy) in [(1, 5), (1, 6), (1, 7), (1, 8), (0, 6), (0, 7), (0, 8),
                            (2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8),
                            (4, 6), (4, 7), (4, 8)]:
                return 1
            if (sx, sy) in [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10)]:
                return 3

        print state_int
        print option
        print "!!!!!"
        raise ValueError("Unexpected state.")

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size - 1:
            return 0
        if sy < self.grid_size - 1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1

    def generate_option_option_trajectories(self, trajectories, n_trajectories, option_policy, action_policy, random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        generated_trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            option = self.options[0]
            path_idx = np.random.choice(len(trajectories[option["id"]]), 1)
            local_path = trajectories[option["id"]][path_idx][0]
            acc_reward = np.sum(local_path[:, 2])
            states = local_path[:, 0]
            state_int = local_path[-1, 0]
            while self.int_to_point(state_int) != option["termination"]:
                new_point = self.generate_intra_option_trajectories(
                              1, 2, action_policy, option,
                              predefined_start=self.int_to_point(state_int))[0][1]
                states = np.append(states, new_point[0])
                state_int = states[-1]

            sx, sy = self.int_to_point(states[-1])
            trajectory.append((states, option["id"], acc_reward))
            for _ in range(len(trajectories[0][0])/2 - 1):
                states = []
                if rn.random() < self.wind:
                    # save this for generalised variant
                    # _available_options = [opt['id'] for opt in self.options if self.point_to_int((sx, sy)) in self.rooms[opt["room"]]]
                    _available_options = [opt['id'] for opt in self.options if (sx, sy) == opt["init_set"]]
                    option = self.options[np.random.choice(_available_options, 1)[0]]
                else:
                    # Follow the given policy.
                    option = self.options[option_policy(option["id"])]

                if (sx, sy) != self.options[2]["termination"]:
                    path_idx = np.random.choice(len(trajectories[option["id"]]), 1)
                    local_path = trajectories[option["id"]][path_idx][0]
                    acc_reward = np.sum(local_path[:, 2])
                    states = local_path[:, 0]
                    state_int = local_path[-1, 0]
                    while self.int_to_point(state_int) != option["termination"]:
                        new_point = self.generate_intra_option_trajectories(
                            1, 2, action_policy, option,
                            predefined_start=self.int_to_point(state_int))[0][1]
                        states = np.append(states, new_point[0])
                        acc_reward += new_point[2]
                        state_int = states[-1]

                    sx, sy = self.int_to_point(states[-1])
                    trajectory.append((states, option["id"], acc_reward))

            generated_trajectories.append(trajectory)

        return np.array(generated_trajectories)


    def generate_intra_option_trajectories(self, n_trajectories, trajectory_length, policy,
                                           option, predefined_start=None, random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            elif predefined_start != None:
                sx, sy = predefined_start
            else:
                if option["id"] == 0 or option["id"] == 7:
                    sx, sy = self.int_to_point(56)
                elif option["id"] == 2 or option["id"] == 1:
                    sx, sy = self.int_to_point(27)
                elif option["id"] == 3 or option["id"] == 4:
                    sx, sy = self.int_to_point(74)
                else:
                    sx, sy = self.int_to_point(104)

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)), option["id"])]

                if (((sx + action[0],
                      sy + action[1]) == option["termination"]) or (
                          option["min"][0] < sx + action[0] < option["max"][0] and
                          option["min"][1] < sy + action[1] < option["max"][1])
                   ):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int) # do not hardcode option
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)

    def generate_trajectories(self, n_trajectories, trajectory_length, policy,
                              random_start=False):
        """
        Generate n_trajectories trajectories with length trajectory_length,
        following the given policy.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        random_start: Whether to start randomly (default False). bool.
        -> [[(state int, action int, reward float)]]
        """

        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0

            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    action = self.actions[rn.randint(0, 4)]
                else:
                    # Follow the given policy.
                    action = self.actions[policy(self.point_to_int((sx, sy)))]

                if (0 <= sx + action[0] < self.grid_size and
                        0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int)# do not hardcode option
                trajectory.append((state_int, action_int, reward))

                sx = next_sx
                sy = next_sy

            trajectories.append(trajectory)

        return np.array(trajectories)
