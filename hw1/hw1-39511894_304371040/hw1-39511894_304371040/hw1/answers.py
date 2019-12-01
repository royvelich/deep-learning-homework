r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Let's look at the two extreme cases:

 - When k=1, we get complete overfitting to the training set.
 - When k="number of samples in training set", the overfitting is reduced to minimum, the model will return the same class for all unseen data.

This suggest that typical k should not be too close to one of the extreme points, and is probably depends on the density of the traning set, which in turn should be similar to the density of the true distribution. That's why in practice we use cross-validation to determine k.

"""

part2_q2 = r"""
**Your answer:**


 1. This method will select the model which best overfits the whole training data, while in cross-validation we only overfit a portion of the training data.
 2. Doing so is equivalent to performing cross-validation on the test set - that is, we create a bias of the model towards the test set, and this is forbidden.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Delta has no effect on the gradient calculation. That is, if the data is linearly separable, then the gradient descent iterations will always converge to the same minima.

"""

part3_q2 = r"""
**Your answer:**


The linear model is learning the average pattern of the images of the same class from the whole dataset.
This is different from KNN in the sense that in KNN we are learning the pattern of the K most similar samples of the same class.

"""

part3_q3 = r"""
**Your answer:**


 1. Our learning rate is good, since we can see that at the first epochs the loss of the training and validation set drops fast and the accuracy of the validation set climbs fast, then they all slowly flattening as the epochs progress.
On the other hand, after expiramenting with the learning rate value, we have noticed that for slow learning rates, the loss of the validation and training set will drop quite slowly, and the accuracy of the vlidation set will have close to linear climbing behavior.
After a sharp increase to the learning rate, we have noticed that we have sharp spikes in both the the loss and accuracy, which means that the gradient cannot converge to a local minima, and keep "jumping" around.
 2. Based on the graph of the training and test set accuracy, we can see that our model slightly underfitted to the training set, since the accuracy curve slope is still positive at the last epochs.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**


 1. The ideal pattern for a residual plot is a straight horizontal line that passes through 0, with very narrow boundaries.
 2. We can clearly see that the final residual plot has much narrower bounderies in which almost all data point are located, there are only few samples that are far away from the line.

"""

part4_q2 = r"""
**Your answer:**


Since we can analytically fit our data (that is, data fitting takes a single "iteration"), the number of time we fit our data equals to the number of combinations of different hyper parameters we want to test times the number of cross validation folds. In our case, we have 3 different values for "degree", 20 different values for lambda, and 3 folds. Therefore, we fit the data 180 times.

"""

# ==============
