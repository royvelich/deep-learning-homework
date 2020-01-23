r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 32
    hypers['h_dim'] = 128
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.3
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "To be or not to be?"
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences instead of training on the whole text in order to avoid vanishing or exploding
gradients, and because we usually cannot fit the whole corpus into the GPU memory.

"""

part1_q2 = r"""
**Your answer:**

It is possible that the generated text shows memory longer than the sequence length because we are
propagating the hidden state between sequences, and that way we keep the context of previous characters seen.

"""

part1_q3 = r"""
**Your answer:**

We are not shuffling the order of batches when training, because we want to keep semantic context of the
hidden layer between adjacent sequences.

"""

part1_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['h_dim'] = 128
    hypers['z_dim'] = 10
    hypers['x_sigma2'] = 5
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The hyper-parameter σ2 governs the relative strength of the two lose function terms. The smaller is σ2, the more the
regression term dominates over the regularization term.

"""

part2_q2 = r"""
**Your answer:**

1. The first term can be thought of as a data fitting term like in regression, demanding that the encoder-decoder
combination is nearly an identity map, that is, we it measures how good the decoder reconstruct the input samples.
The second term applies regularization on the output of the encoder in the latent space. That is, it measures the
divergence of the latent space distribution from the standard normal distribution.

2. The smaller the KL loss term, the closer the latent-space distribution to the standard normal distribution.

3. The benefit of this is that it allows us to sample a random vector from the known standard normal distribution
in order to generate a sample which lays on the higher-dimension manifold from which we have taken our training
samples.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['z_dim'] = 6
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.2
    hypers['discriminator_optimizer']['type'] = 'Adam'
    hypers['discriminator_optimizer']['lr'] = 0.00001
    hypers['generator_optimizer']['type'] = 'Adam'
    hypers['generator_optimizer']['lr'] = 0.001
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

When we train the *discriminator*, we want to supply it with a real sample and a fake sample. Therefore, in order to
create the fake sample, we use the generator to generate one. However, after the forward discriminator forward pass,
we don't want gradients to pass back through the generator, since we want to train ONLY the discriminator. Therefore,
we detach the generator's output from the graph in order to prevent gradients to pass back through it.
When we want to train the generator, we do want gradients to pass through it during the backward pass, in order for it
to learn how to generator realistic fake samples.

"""

part3_q2 = r"""
**Your answer:**

1. When we train a GAN, our criteria to stop training should be based both on the the loss values of the
generator AND the discriminator. That is because we want both of them to be of equal "strength". We want that the
probability of the discriminator to distinguish between a real sample a fake sample to be 50%, that is, that it will be
equivalent to a coin flip.

2. If the discriminator loss remains at a constant value while the generator loss decreases, it means that the generator
is still learning how generate realistic samples, but still not as good as is required to fool the discriminator.

"""

part3_q3 = r"""
**Your answer:**

The main difference between the GAN and the VAE results, is that the VAE results seem to be much more blurry than the
GAN results. That is, the GAN has the potential to produce sharp realistic images. That is because the GAN implicitly
receives an additional feedback from the discriminator which directs it to create realistic samples.

"""

# ==============


