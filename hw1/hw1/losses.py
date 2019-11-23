import abc
import torch
import hw1.transforms as hw1tf

class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        indices = torch.Tensor(y.to(dtype=torch.float32).unsqueeze(1)).to(dtype=int)
        ground_truth_scores = torch.gather(x_scores, 1, indices).expand(*x_scores.size())

        rows = torch.arange(0, x_scores.size(0)).unsqueeze(0).to(dtype=int)
        columns = y.unsqueeze(0).to(dtype=int)

        M = x_scores.clone()
        M[rows, columns] = M[rows, columns].clone() - self.delta
        M = torch.sub(M.clone() + self.delta, ground_truth_scores)
        M = torch.max(M.clone(), torch.zeros_like(x_scores.clone()))

        loss = torch.sum(M) / float(x_scores.size(0))

        self.grad_ctx['M'] = M
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['rows'] = rows
        self.grad_ctx['columns'] = columns
        # TODO: Save what you need for gradient calculation in self.grad_ctx

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        M = self.grad_ctx['M']
        x = self.grad_ctx['x']
        y = self.grad_ctx['y']
        rows = self.grad_ctx['rows']
        columns = self.grad_ctx['columns']

        bias_trick = hw1tf.BiasTrick()
        x = bias_trick(x)

        M2 = (M > 0).to(dtype=torch.float32)

        # print(M2)

        k = M2.sum(dim=1)
        # print(k)

        M2[rows, columns] = M2[rows, columns].clone() * (-k)

        print(y)
        print(M2)

        grad = torch.mm(x.transpose(0,1), M2) / M.size(0)
        return grad
