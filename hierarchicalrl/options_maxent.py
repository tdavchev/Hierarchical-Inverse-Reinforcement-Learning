"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import sys
sys.path.append("/Users/todordavchev/Documents/temp/")

import matplotlib.pyplot as plt
import numpy as np

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
    trajectory_length = grid_size

    walls = [
        (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
        (0, 5), (2, 5), (3, 5), (4, 5),
        (6, 6), (7, 6), (9, 6), (10, 6)
        ]

    options = [
        {'init_set': (5, 2), 'termination': (1, 5), 'room': 0},
        {'init_set': (5, 2), 'termination': (8, 6), 'room': 0},
        {'init_set': (8, 6), 'termination': (5, 2), 'room': 1},
        {'init_set': (8, 6), 'termination': (5, 9), 'room': 1},
        {'init_set': (5, 9), 'termination': (8, 6), 'room': 2},
        {'init_set': (5, 9), 'termination': (1, 5), 'room': 2},
        {'init_set': (1, 5), 'termination': (5, 9), 'room': 3},
        {'init_set': (1, 5), 'termination': (5, 2), 'room': 3},
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
            66, 67, 68, 69, 70,
            77, 78, 79, 80, 81,
            88, 89, 90, 91, 92,
            99, 100, 101, 102, 103,
            110, 111, 112, 113, 114,
            56, 104
        ],
        [
            83, 84, 85, 86, 87,
            94, 95, 96, 97, 98,
            105, 106, 107, 108, 109,
            116, 117, 118, 119, 120,
            104, 74
        ]
        ]
    g_world = options_gridworld.Large_Gridworld(grid_size, walls, options, rooms, wind, discount)
    trajectories = g_world.generate_intra_option_trajectories(n_trajectories,
                                                              trajectory_length,
                                                              g_world.intra_option_optimal_policy)
    feature_matrix = g_world.feature_matrix()
    ground_r = np.array([g_world.reward(s) for s in range(g_world.n_states)])
    reward = maxent.irl([rooms[options[0]["room"]]], feature_matrix[0], g_world.n_actions, discount,
                        g_world.improved_transition_probability[0], trajectories,
                        epochs, learning_rate)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(11, 0.01, 20, 200, 0.01)
