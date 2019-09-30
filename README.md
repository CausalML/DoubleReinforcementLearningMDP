# DoubleReinforcementLearningMDP

This repository contains the code for replicate the experiments from the paper 
### "Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes"
- https://arxiv.org/abs/1908.08526

## Experiments in Section 5.1

The relevant code is in the subdirectory `exp5_1`. 
* `toytoy.py` runs the experiment with the in-sample variant of the estimators.
* `toytoy2.py` runs the experiment with the samples-splitting variant of the estimators.

For example, to run 10 parallel replications, one can run the command `seq 10 | xargs -L 1 -P 10 ./toytoy.sh`

## Experiments in Section 5.2

The relevant code is in the subdirectory `exp5_2`. 
