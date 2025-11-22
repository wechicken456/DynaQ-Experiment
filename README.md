# DynaQ-Experiment

Dyna-Q implementation in [scripts](./scripts/dynaq/) that is currently being tested on Gymnasium's LunarLander-v2 environment.


# Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

Then, run:

```
uv sync
```


# References used for my implementation


[Google DeepMind March 2016] (https://arxiv.org/pdf/1603.00748v1).

- More pseudocode and hyperparameter information [here](https://fernandoperezc.github.io/Advanced-Topics-in-Machine-Learning-and-Data-Science/Lohmann.pdf).



# Roadblocks & Lessons

## Follow past research papers to recreate results``

## Do NOT use the same seed to reset the environment

Cause then the agent will always start at the same state and won't learn other patterns.


## Do NOT try to modularize too much

Cause we experiment and edit lots of things so if we modularize everything we have to change so many different files & code.




