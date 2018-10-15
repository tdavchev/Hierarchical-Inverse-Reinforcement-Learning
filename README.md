# Hierarchical Inverse Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.555999.svg)](https://doi.org/10.5281/zenodo.555999)

Extends [M. Alger's](https://doi.org/10.5281/zenodo.555999) implementation of selected inverse reinforcement learning (IRL). A summary report of my extension is available [here](https://www.overleaf.com/read/mkkfqgpnbvnr). His final report is available [here](http://matthewja.com/pdfs/irl.pdf) and describes the implemented algorithms.

If you use this code in your work, you can cite it as follows:
```bibtex
@misc{alger16,
  author       = {Todor Davchev},
  title        = {Hierarchical Inverse Reinforcement Learning},
  year         = 2017
}
```
If you are only interested in the IRL aspect of this project, you can find at [Alger's repo](https://github.com/MatthewJA/Inverse-Reinforcement-Learning)
## Algorithms implemented

- Linear programming IRL. From Ng & Russell, 2000. Small state space and large state space linear programming IRL.
- Maximum entropy IRL. From Ziebart et al., 2008.
- Deep maximum entropy IRL. From Wulfmeier et al., 2015; original derivation.
- Hierarchical MaxEnt IRL.

Additionally, the following MDP domains are implemented:
- Gridworld (Sutton, 1998)
- Extended Gridworld with options (Sutton, 1998)
- Objectworld (Levine et al., 2011)

## Requirements
- NumPy
- SciPy
- CVXOPT
- Theano
- MatPlotLib (for examples)
