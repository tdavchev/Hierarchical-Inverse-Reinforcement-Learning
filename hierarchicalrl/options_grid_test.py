"""
Unit tests for the gridworld MDP.

Todor Davchev, 2017
t.b.davchev@ed.ac.uk
"""

import unittest

import numpy as np
import numpy.random as rn

import options_grid_world as gridworld


def make_random_gridworld():
    grid_size = rn.randint(2, 15)
    wind = rn.uniform(0.0, 1.0)
    discount = rn.uniform(0.0, 1.0)
    return gridworld.Gridworld(grid_size, wind, discount)


class TestTransitionProbability(unittest.TestCase):
    """Tests for Gridworld.transition_probability."""

    # def test_sums_to_one(self):
    #     """Tests that the sum of transition probabilities is approximately 1."""
    #     # This is a simple fuzz-test.
    #     for _ in range(40):
    #         gw = make_random_gridworld()
    #         self.assertTrue(
    #             np.isclose(gw.transition_probability.sum(axis=2), 1).all(),
    #             'Probabilities don\'t sum to 1: {}'.format(gw))

    def test_manual_sums_to_one(self):
        """Tests issue #1 on GitHub."""
        walls = [
            (5, 0), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 10),
            (0, 5), (2, 5), (3, 5), (4, 5),
            (6, 6), (7, 6), (9, 6), (10, 6)
        ]
        gw = gridworld.Large_Gridworld(11, walls, 0.3, 0.2)
        self.assertTrue(
            np.isclose(gw.options_transition_probability.sum(axis=2), 1).all())

        # take out all walls since their probabilities == 0
        bb = gw.improved_transition_probability.sum(axis=3)
        aa = gw.transition_probability.sum(axis=2)
        self.assertTrue(
            np.isclose([x for i, x in enumerate(aa) if x.all() != 0.], 1).all())

if __name__ == '__main__':
    unittest.main()