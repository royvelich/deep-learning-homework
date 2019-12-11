import torch
import itertools as it
import torch.nn as nn
import math

class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes: int, channels: list,
                 pool_every: int, hidden_dims: list):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.P = self.pool_every
        self.N = len(self.channels)
        self.M = len(self.hidden_dims)
        self.quotient = int(self.N / self.P)
        self.reminder = int(math.fabs(math.remainder(self.N, self.P)))

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ReLU)*P -> MaxPool]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs should exist at the end, without a MaxPool after them.
        # ====== YOUR CODE: ======
        k = 0
        for i in range(self.quotient):
            for j in range(self.P):
                layers.append(nn.Conv2d(in_channels=in_channels if k == 0 else self.channels[k - 1], out_channels=self.channels[k], kernel_size=(3, 3), padding=1))
                layers.append(nn.ReLU())
                k += 1
            layers.append(torch.nn.MaxPool2d(2))

        for i in range(self.reminder):
            layers.append(nn.Conv2d(in_channels=self.channels[k - 1], out_channels=self.channels[k], kernel_size=(3, 3), padding=1))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        #  (Linear -> ReLU)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        self.flatten_features_size = int(in_w / math.pow(2, self.quotient)) * int(in_h / math.pow(2, self.quotient)) * self.channels[(self.P * self.quotient + self.reminder) - 1]
        for i in range(self.M):
            layers.append(nn.Linear(self.flatten_features_size if i == 0 else self.hidden_dims[i - 1], self.hidden_dims[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_dims[self.M - 1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor.forward(x)
        features = features.reshape(-1, self.flatten_features_size)
        out = self.classifier.forward(features)
        # ========================
        return out


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(self, in_channels: int, channels: list, kernel_sizes: list,
                 batchnorm=False, dropout=0.):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
        convolution in the block. The length determines the number of
        convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
        be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
        convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
        Zero means don't apply dropout.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  the main_path, which should contain the convolution, dropout,
        #  batchnorm, relu sequences, and the shortcut_path which should
        #  represent the skip-connection.
        #  Use convolutions which preserve the spatial extent of the input.
        #  For simplicity of implementation, we'll assume kernel sizes are odd.
        # ====== YOUR CODE: ======
        main_layers = []
        for i in range(len(channels)):
            main_layers.append(torch.nn.Conv2d(in_channels=in_channels if i == 0 else channels[i - 1], out_channels=channels[i], kernel_size=kernel_sizes[i], padding=int((kernel_sizes[i] - 1) / 2)))
            if i != (len(channels) - 1):
                if batchnorm is True:
                    main_layers.append(torch.nn.BatchNorm2d(num_features=channels[i]))
                if dropout > 0:
                    main_layers.append(torch.nn.Dropout2d(dropout))
                main_layers.append(torch.nn.ReLU())
        self.main_path = nn.Sequential(*main_layers)

        shortcut_layers = []
        output_channels = channels[len(channels) - 1]
        if in_channels != output_channels:
            shortcut_layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1, bias=False))

        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ReLU)*P -> MaxPool]*(N/P)
        #   \------- SKIP ------/
        #  Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        #  Pooling to reduce dimensions after every P convolutions.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ReLUs (with a skip over them) should exist at the end,
        #  without a MaxPool after them.
        # ====== YOUR CODE: ======
        k = 0
        for i in range(self.quotient):
            layers.append(ResidualBlock(in_channels=in_channels if k == 0 else self.channels[k - 1], channels=self.channels[k:k+self.P], kernel_sizes=[3]*self.P))
            layers.append(torch.nn.MaxPool2d(2))
            k += self.P

        if self.reminder > 0:
            layers.append(ResidualBlock(in_channels=self.channels[k - 1], channels=self.channels[k:k + self.reminder], kernel_sizes=[3] * self.reminder))
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every,
                 hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every,
                         hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======

    # ========================
