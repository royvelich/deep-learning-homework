r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.05
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1
    lr_vanilla = 0.1
    lr_momentum = 0.008
    lr_rmsprop = 0.02
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
  1. We can clearly see that the graphs of no-dropout (dropout = 0) vs dropout=0.4 visualize the typical characteristics of 
  a network that overfits the training set versus a network that generalizes well for unseen samples.
  For example:
    1. We can see that the test accuracy of dropout=0.4 is higher than the test accuracy of dropout=0.
    2. We can see that the test loss of dropout=0 starts to rise again at epoch=15 while the test loss of dropout=0.4 
    keeps decreasing slowly towards convergence (similar to dropout=0.8).
    3. We can see that at dropout=0 the train loss at the final epoch is almost 0, which means severe overfitting.
  2. We can clearly see that the test accuracy of dropout=0.4 is much higher than the test accuracy of
  dropout=0.8 (which is even lower than dropout=0), and this is because when we use very high dropout, which is
  equivalent to high regularization measure, the network is prevented from getting exposed to the real training data,
  and thus fails to learn effectively.
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible.
The accuracy of a dataset is evaluated by the maximal value of the softmax layer. It is not dependent on how high the
softmax output is in absolute terms, only relative terms. For example, we can have a classifier with 3 different
classes - A, B and C. We can have two given samples which will be classified the same but with different losses. 
For example, SAMPLE1 = [A=0.9, B=0.1, C=0] and SAMPLE2 = [A=0.4,B=0.3,C=0.3]. In both cases we will get the same
classification, however, we get that loss(SAMPLE1) = 0.04 and loss(SAMPLE2) = 0.39. That is, we see that the accuracy
can be sustained or even increase as the probabilities of the different classes are shifted towards each other as the
epochs progress.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
  1. We can clearly see that as the depth deepens the accuracy drops. The depth that produced the best
results is L=4 K=64, and this is because the gradients are still viable at this depth.
  2. At L=16 the network wasn't trainable at all (loss stuck at constant level). The cause for this is the phenomena
of vanishing gradients. To resolve this issue, we can use either 'skip-connections' to let gradients from deeper
regions jump downwards, or we can use regularization techniques. 

"""

part3_q2 = r"""
**Your answer:**
We can clearly see that also in this experiment the best results are achieved at L=4 K=64. That means that in simple
CNN such as ours, increasing the number of filters doesn't compensate for vanishing gradients caused by depth.
"""

part3_q3 = r"""
**Your answer:**
This experiment emphasis further that for simple CNN the depth plays a major role in the training capability of the
network, since we use a varying number of filters at each block. We can clearly see that for depths L=2,4 we get very similar
results, while a sharp drop occurs as the depth increases, up to the point of untraiable network.
"""

part3_q4 = r"""
**Your answer:**
Here we can clearly see that the 'skip-connections' of the residual blocks are drastically improving the ability of the
network to be trainable even at very high depths. As we can see from the results, the network is trainable even at L=32.
"""

part3_q5 = r"""
**Your answer:**
  1. Our modifications were to add dropout regularization, batch normalization and use a kernel size of 5.
  2. We can clearly see that our network is trainable at all configurations and that we have achieved better test
  accuracy - that is, thanks to regularization, we have prevented overfitting the training dataset, and improved the
  generalization capabilities of our network. 
"""
# ==============
