import re

import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    text = sorted(''.join(set(text)))
    char_to_idx = {}
    idx_to_char = {}
    for i in range(len(text)):
        idx_to_char[i] = text[i]
        char_to_idx[text[i]] = i
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = text
    for char in chars_to_remove:
        text_clean = text_clean.replace(char, '')

    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tesnsor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    N = len(text)
    D = len(char_to_idx.keys())
    indices = []
    for c in text:
        indices.append(char_to_idx[c])
    result = torch.zeros(N, D)
    result[range(N), indices] = 1
    # ========================
    return result.to(dtype=torch.int8)


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    indices = (embedded_text != 0).nonzero()[:, 1]
    result = ''
    for i in indices:
        result = result + idx_to_char[int(i)]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int,
                              device='cpu'):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    S = seq_len
    N = (len(text) - 1) // seq_len
    text_length = N * seq_len
    samples_text = text[0:text_length]
    samples_embedded_text = chars_to_onehot(samples_text, char_to_idx)
    V = samples_embedded_text.shape[1]
    samples = samples_embedded_text.reshape(N, S, V)

    labels_text = text[1:text_length+1]
    labels_embedded_text = chars_to_onehot(labels_text, char_to_idx)
    indices = (labels_embedded_text != 0).nonzero()[:, 1]
    labels = indices.reshape(N, S)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled_y = y / temperature
    m = torch.nn.Softmax(dim=dim)
    result = m(scaled_y)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        for i in range(n_chars - len(start_sequence)):
            onehot_tensor = chars_to_onehot(out_text, char_to_idx).unsqueeze(0).to(device)
            output, hidden = model.forward(onehot_tensor.to(dtype=torch.float))
            next_char_scores = output[0, len(start_sequence) + i - 1, :]
            next_char_prob = hot_softmax(next_char_scores, temperature=T)
            idx = int(torch.multinomial(next_char_prob, 1))
            next_char = idx_to_char[idx]
            out_text = out_text + next_char
    # ========================
    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents  one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of indices is takes, samples in the same index of
        #  adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        idx = []
        num_of_batches = len(self.dataset) // self.batch_size
        for i in range(num_of_batches):
            for j in range(self.batch_size):
                idx.append(j * self.batch_size + i)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        for i in range(n_layers):
            current_in_dim = h_dim
            if i == 0:
                current_in_dim = in_dim

            z1 = torch.nn.Linear(current_in_dim, h_dim, bias=False)
            z2 = torch.nn.Linear(h_dim, h_dim)
            z_sig = torch.nn.Sigmoid()

            r1 = torch.nn.Linear(current_in_dim, h_dim, bias=False)
            r2 = torch.nn.Linear(h_dim, h_dim)
            r_sig = torch.nn.Sigmoid()

            g1 = torch.nn.Linear(current_in_dim, h_dim, bias=False)
            g2 = torch.nn.Linear(h_dim, h_dim)
            g_tanh = torch.nn.Tanh()

            d = torch.nn.Dropout2d(dropout)

            self.add_module('z1' + str(i), z1)
            self.add_module('z2' + str(i), z2)
            self.add_module('z_sig' + str(i), z_sig)

            self.add_module('r1' + str(i), r1)
            self.add_module('r2' + str(i), r2)
            self.add_module('r_sig' + str(i), r_sig)

            self.add_module('g1' + str(i), g1)
            self.add_module('g2' + str(i), g2)
            self.add_module('g_tanh' + str(i), g_tanh)

            self.add_module('d' + str(i), d)

            params = {
                'z1': z1,
                'z2': z2,
                'z_sig': z_sig,
                'r1': r1,
                'r2': r2,
                'r_sig': r_sig,
                'g1': g1,
                'g2': g2,
                'g_tanh': g_tanh,
                'd': d
            }

            self.layer_params.append(params)

        self.Y = torch.nn.Linear(h_dim, out_dim)
        self.add_module('Y', self.Y)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device))
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        if next(self.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        input_per_t = []
        for i in range(self.n_layers+1):
            input_per_t.append([])

        for i in range(seq_len):
            input_per_t[0].append(layer_input[:, i, :])

        hidden_per_t = []
        for i in range(self.n_layers):
            hidden_per_t.append([])
            hidden_per_t[i].append(layer_states[i])

        for i in range(self.n_layers):
            params = self.layer_params[i]
            z1 = params['z1']
            z2 = params['z2']
            z_sig = params['z_sig']

            r1 = params['r1']
            r2 = params['r2']
            r_sig = params['r_sig']

            g1 = params['g1']
            g2 = params['g2']
            g_tanh = params['g_tanh']

            d = params['d']
            for j in range(seq_len):
                current_hidden = hidden_per_t[i][j].to(device)
                current_input = input_per_t[i][j].to(device)
                z_act = z_sig(z1(current_input) + z2(current_hidden))
                r_act = r_sig(r1(current_input) + r2(current_hidden))
                g_act = g_tanh(g1(current_input) + g2(current_hidden * r_act))
                h = (z_act * current_hidden + (1 - z_act) * g_act).to(device)
                hidden_per_t[i].append(h.to(device))
                input_per_t[i+1].append(d(h.to(device)))

        last_output_layer = input_per_t[self.n_layers]
        output_sequence = []
        for i in range(seq_len):
            output_sequence.append(self.Y(last_output_layer[i].to(device)))

        hidden_sequence = []
        for i in range(self.n_layers):
            hidden_sequence.append(hidden_per_t[i][seq_len-1].to(device))

        for i in range(seq_len):
            if i == 0:
                layer_output = output_sequence[0].to(device)
            else:
                layer_output = torch.cat([layer_output.to(device), output_sequence[i].to(device)], dim=1)

        for i in range(self.n_layers):
            if i == 0:
                hidden_state = hidden_sequence[0].to(device)
            else:
                hidden_state = torch.cat([hidden_state.to(device), hidden_sequence[i].to(device)], dim=1)

        layer_output = layer_output.reshape(batch_size, seq_len, self.out_dim)
        hidden_state = hidden_state.reshape(batch_size, self.n_layers, self.h_dim)
        # ========================
        return layer_output, hidden_state
