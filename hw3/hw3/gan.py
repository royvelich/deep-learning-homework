from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        modules.append(torch.nn.Conv2d(in_channels=in_size[0], out_channels=8, kernel_size=(16, 16)))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.Dropout2d(0.5))
        modules.append(torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(16, 16)))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.Dropout2d(0.5))
        modules.append(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(16, 16)))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.Dropout2d(0.5))
        modules.append(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(16, 16)))
        self.cnn = nn.Sequential(*modules)
        self.flat_dim = 64*4*4
        self.affine = nn.Linear(self.flat_dim, 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        features = self.cnn(x)
        features_flat = features.reshape(x.shape[0], self.flat_dim)
        y = self.affine(features_flat)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.featuremap_size = featuremap_size
        self.affine = nn.Linear(z_dim, featuremap_size*featuremap_size)
        modules = []
        modules.append(torch.nn.BatchNorm2d(1))
        modules.append(torch.nn.Upsample(scale_factor=2))
        modules.append(torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.BatchNorm2d(128, 0.8))
        modules.append(torch.nn.Upsample(scale_factor=2))
        modules.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.BatchNorm2d(128, 0.8))
        modules.append(torch.nn.Upsample(scale_factor=2))
        modules.append(torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        modules.append(nn.LeakyReLU())
        modules.append(torch.nn.BatchNorm2d(128, 0.8))
        modules.append(torch.nn.Upsample(scale_factor=2))
        modules.append(torch.nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1))
        modules.append(nn.Tanh())

        # modules.append(torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(16, 16)))
        # modules.append(nn.LeakyReLU())
        # modules.append(torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(16, 16)))
        # modules.append(nn.LeakyReLU())
        # modules.append(torch.nn.ConvTranspose2d(in_channels=16, out_channels=out_channels, kernel_size=(16, 16)))

        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        if not with_grad:
            z = torch.randn(n, self.z_dim).to(device)
            samples = self.forward(z).detach()
        else:
            z = torch.randn(n, self.z_dim).to(device)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        features = self.affine(z)
        features_reshaped = features.reshape(z.shape[0], 1, self.featuremap_size, self.featuremap_size)
        x = self.cnn(features_reshaped)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    data_loss_fn = torch.nn.BCEWithLogitsLoss()
    generated_loss_fn = torch.nn.BCEWithLogitsLoss()
    data_labels_noise = ((torch.rand(*y_data.shape) * label_noise) - (label_noise/2)).cuda()
    generated_labels_noise = ((torch.rand(*y_data.shape) * label_noise) - (label_noise/2)).cuda()

    if data_label == 1:
        data_labels = torch.ones_like(y_data).cuda() + data_labels_noise
        generated_labels = torch.zeros_like(y_generated).cuda() + generated_labels_noise
    else:
        data_labels = torch.zeros_like(y_data) + data_labels_noise
        generated_labels = torch.ones_like(y_generated) + generated_labels_noise

    loss_data = data_loss_fn(y_data.cuda(), data_labels.cuda())
    loss_generated = generated_loss_fn(y_generated.cuda(), generated_labels.cuda())
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.BCEWithLogitsLoss()

    if data_label == 1:
        labels = torch.ones_like(y_generated)
    else:
        labels = torch.zeros_like(y_generated)

    loss = loss_fn(y_generated, labels)
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    n = x_data.shape[0]
    dsc_optimizer.zero_grad()
    x_generated = gen_model.sample(n, with_grad=False)
    real_scores = dsc_model(x_data)
    fake_scores = dsc_model(x_generated)
    dsc_loss = dsc_loss_fn(real_scores, fake_scores)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    # for i in range(2):
    gen_optimizer.zero_grad()
    x_generated2 = gen_model.sample(n, with_grad=True)
    fake_scores2 = dsc_model(x_generated2)
    gen_loss = gen_loss_fn(fake_scores2)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    # saved_state = dict(dsc_losses=dsc_losses,
    #                    gen_losses=gen_losses,
    #                    model_state=gen_model.state_dict())
    # torch.save(saved_state, checkpoint_file)
    # saved = True
    # ========================

    return saved
