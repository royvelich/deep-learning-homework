import torch
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss
import hw1.transforms as hw1tf



class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = torch.randn(n_features, n_classes) * weight_std

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred = torch.Tensor(x.size(0))
        S = torch.mm(x, self.weights)
        class_scores = S
        for i in range(x.size(0)):
            current_class_scores = S[i, :]
            y_pred[i] = current_class_scores.argmax()

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = float(torch.sum(y == y_pred)) / float(y.size(0))

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):
            total_correct = 0
            total_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            train_batches = 0
            for idx, (images, classes) in enumerate(dl_train):
                y_pred, class_scores = self.predict(images)
                current_loss = loss_fn.loss(images, classes, class_scores, y_pred)
                current_accuracy = self.evaluate_accuracy(classes, y_pred)
                total_loss = total_loss + current_loss
                total_correct = total_correct + current_accuracy
                self.weights = self.weights - learn_rate*loss_fn.grad() - weight_decay * self.weights
                train_batches = train_batches + 1

            total_correct_valid = 0
            total_loss_valid = 0
            valid_batches = 0
            for idx, (images_valid, classes_valid) in enumerate(dl_valid):
                y_pred_valid, class_scores_valid = self.predict(images_valid)
                current_loss_valid = loss_fn.loss(images_valid, classes_valid, class_scores_valid, y_pred_valid)
                current_accuracy_valid = self.evaluate_accuracy(classes_valid, y_pred_valid)
                total_loss_valid = total_loss_valid + current_loss_valid
                total_correct_valid = total_correct_valid + current_accuracy_valid
                valid_batches = valid_batches + 1

            valid_res.loss.append(total_loss_valid / valid_batches)
            valid_res.accuracy.append(total_correct_valid / valid_batches)

            train_res.loss.append(total_loss / train_batches)
            train_res.accuracy.append(total_correct / train_batches)

            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        print(self.weights.size())

        rows = self.weights.size()[0]
        cols = self.weights.size()[1]


        w_images = self.weights[list(range(1, rows))]
        w_images = w_images.transpose(-1, 0)

        w_images = w_images.reshape(cols, img_shape[0], img_shape[1], img_shape[2])

        return w_images


def hyperparams():
    hp = dict(weight_std=0., learn_rate=0., weight_decay=0.)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.

    hp['weight_std'] = 1
    hp['learn_rate'] = 0.2
    hp['weight_decay'] = 0.01

    return hp
