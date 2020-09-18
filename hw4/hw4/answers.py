r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,#32 #64 #sh16
              gamma=0.99,#0.99 #0.99 #sh0.99
              beta=0.5,#0.5 #0.5 #sh0.5
              learn_rate=1e-3,#1e-3 #6e-3 #sh1e-3
              eps=1e-8#1e-8 #1e-8 #sh0.0003
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=32,
              gamma=0.8,
              beta=0.3,
              learn_rate=1e-2,
              eps=1e-7,
              )
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=64,#32
              gamma=0.99,#0.99
              delta=0.01,#1.
              learn_rate=1e-3,#1e-3
              eps=0.0003,#1e-8
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(batch_size=48,
              gamma=0.8,
              beta=1.,
              delta=0.1,
              learn_rate=1e-2,
              eps=1e-8,
              )
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
