"""
Run maximum entropy inverse reinforcement learning on the options gridworld MDP.

Todor Davchev, 2017
t.b.davchev@ed.ac.uk
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

import sdp_maxent as maxent
import options_grid_world as options_gridworld


def main(grid_size, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size/2

    walls = [
        (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
        (0, 5), (2, 5), (3, 5), (4, 5),
        (6, 6), (7, 6), (9, 6), (10, 6)
        ]

    options = [
        {'init_set': (1, 5), 'termination': (5, 2), 'room': 0, 'id': 0,
         "min": (-1, -1), "max": (5, 5)},
        {'init_set': (5, 2), 'termination': (1, 5), 'room': 0, 'id': 1,
         "min": (-1, -1), "max": (5, 5)},
        {'init_set': (5, 2), 'termination': (8, 6), 'room': 1, 'id': 2,
         "min": (5, -1), "max": (11, 6)},
        {'init_set': (8, 6), 'termination': (5, 2), 'room': 1, 'id': 3,
         "min": (5, -1), "max": (11, 6)},
        {'init_set': (8, 6), 'termination': (5, 9), 'room': 2, 'id': 4,
         'min': (5, 6), 'max': (11, 11)},
        {'init_set': (5, 9), 'termination': (8, 6), 'room': 2, 'id': 5,
         'min': (5, 6), 'max': (11, 11)},
        {'init_set': (5, 9), 'termination': (1, 5), 'room': 3, 'id': 6,
         'min': (-1, 5), "max": (5, 11)},
        {'init_set': (1, 5), 'termination': (5, 9), 'room': 3, 'id': 7,
         'min': (-1, 5), "max": (5, 11)}
        ]

    rooms = [
        [
            0, 1, 2, 3, 4,
            11, 12, 13, 14, 15,
            22, 23, 24, 25, 26,
            33, 34, 35, 36, 37,
            44, 45, 46, 47, 48,
            56, 27
        ],
        [
            6, 7, 8, 9, 10,
            17, 18, 19, 20, 21,
            28, 29, 30, 31, 32,
            39, 40, 41, 42, 43,
            50, 51, 52, 53, 54,
            61, 62, 63, 64, 65,
            74, 27
        ],
        [
            83, 84, 85, 86, 87,
            94, 95, 96, 97, 98,
            105, 106, 107, 108, 109,
            116, 117, 118, 119, 120,
            104, 74
        ],
        [
            66, 67, 68, 69, 70,
            77, 78, 79, 80, 81,
            88, 89, 90, 91, 92,
            99, 100, 101, 102, 103,
            110, 111, 112, 113, 114,
            56, 104
        ]
    ]
    g_world = options_gridworld.Large_Gridworld(grid_size, walls, options, rooms, wind, discount)
    trajectories = []
    for opt in options:
        trajectories.append(
            g_world.generate_intra_option_trajectories(
                n_trajectories,
                trajectory_length,
                g_world.intra_option_optimal_policy,
                opt))

    global_trajectories = g_world.generate_option_option_trajectories(
        trajectories, n_trajectories,
        g_world.option_option_optimal_policy,
        g_world.intra_option_optimal_policy)
    feature_matrix = g_world.feature_matrix()
    option_feature_matrix = g_world.o_feature_matrix()
    #the reward needs to be changed not per room but per option..
    ground_r = np.array([g_world.reward(state) for state in range(grid_size**2)])
    ground_opt_r = np.array([g_world.opt_reward(opt) for opt in range(len(options))])
    options_states = [rooms[opts["room"]] for opts in options]
    print("Compute the reward.")
    reward, o_reward = maxent.irl(
        options_states, feature_matrix,
        option_feature_matrix, g_world.n_actions,
        g_world.n_options, discount, g_world.options_transition_probability,
        g_world.improved_transition_probability, trajectories, global_trajectories,
        epochs, learning_rate, g_world.int_to_point, options)
    result = np.zeros((len(options),grid_size**2))
    option_result = np.zeros(8)
    writer = csv.writer(open("results/results.csv", 'w'))
    with open("results/opt_results.csv", 'wb') as csvfile:
        opt_writer = csv.writer(csvfile)
        opt_writer.writerow(o_reward)

    with open("results/results.csv", 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for o in range(len(options)):
            for broi, value in enumerate(options_states[o]):
                result[o][value] = reward[o][broi]
            writer.writerow(result[o])

# plt.savefig('/tmp/test.png')
        # plt.subplot(1, 2, 1)
        # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
        # plt.colorbar()
        # plt.title("Groundtruth reward")
        # plt.subplot(1, 2, 2)
        # plt.pcolor(result[o].reshape((grid_size, grid_size)))
        # plt.colorbar()
        # plt.title("Recovered reward")

    # with open('thefile.csv', 'rb') as f:
    #     data = list(csv.reader(f))
            

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_opt_r.reshape((4, 2)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(o_reward.reshape((4, 2)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(11, 0.01, 20, 200, 0.01)
