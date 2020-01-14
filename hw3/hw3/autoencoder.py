import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        modules.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=(16, 16)))
        modules.append(nn.Tanh())
        # ========================

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        modules.append(torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=32, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(16, 16)))
        modules.append(nn.ReLU())
        modules.append(torch.nn.ConvTranspose2d(in_channels=8, out_channels=out_channels, kernel_size=(16, 16)))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.mean_affine = torch.nn.Linear(in_features=n_features, out_features=z_dim)
        self.sigma_affine = torch.nn.Linear(in_features=n_features, out_features=z_dim)
        self.latent_affine = torch.nn.Linear(in_features=z_dim, out_features=n_features)
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        device = next(self.parameters()).device
        # print(x)
        h = self.features_encoder(x).reshape(1,-1)
        mu = self.mean_affine(h)
        # print(h)
        log_sigma2 = self.sigma_affine(h)
        # print(log_sigma2)
        u = torch.randn(1, self.z_dim).to(device)
        z = mu + u * torch.exp(log_sigma2).to(device)
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.latent_affine(z)
        h_reshaped = h.reshape(*self.features_shape).unsqueeze(0)
        x_rec = self.features_decoder(h_reshaped)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model.
            #  Generate n latent space samples and return their
            #  reconstructions.
            #  Remember that this mean using the model for inference.
            #  Also note that we're ignoring the sigma2 parameter here.
            #  Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #  the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            for i in range(n):
                # u = torch.rand(self.z_dim).to(device)
                # print(self.z_dim)
                # print(self.mean_affine.bias)
                # h = u + self.mean_affine.bias.to(device)
                # h_reshaped = h.reshape(*self.features_shape).unsqueeze(0)
                # x_rec = self.features_decoder(h_reshaped)
                z = torch.randn(self.z_dim).to(device)
                h = self.latent_affine(z)
                h_reshaped = h.reshape(*self.features_shape).unsqueeze(0)
                x_rec = self.features_decoder(h_reshaped)
                samples.append(x_rec.squeeze().to('cpu'))
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    x_shape = x.shape
    z_mu_shape = z_mu.shape
    N = x_shape[0]
    d_x = x_shape[1] * x_shape[2] * x_shape[3]
    d_z = z_mu_shape[1]
    diff = x - xr
    diff_squared_norm = (diff*diff).sum()
    z_sigma2 = torch.exp(z_log_sigma2)
    log_det_sigma = 0
    trace_sigma = 0
    for i in range(N):
        current_sigma = z_sigma2[i, :].squeeze().diag()
        log_det_sigma += torch.log(torch.det(current_sigma))
        trace_sigma += torch.trace(current_sigma)
    z_mu_squared_norm = (z_mu*z_mu).sum()
    factor = 1 / (x_sigma2 * d_x)
    data_loss = (factor * diff_squared_norm) / N
    kldiv_loss = (trace_sigma + z_mu_squared_norm - log_det_sigma) / N - d_z
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
