"""Module with the definition of ConvLSTM classes

Initial work from: https://github.com/ndrplz/ConvLSTM_pytorch
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ConvLSTMCell(nn.Module):
    """
    Convolutional Long Short-Term Memory (ConvLSTM) cell.

    This class implements a ConvLSTM cell, which is a type of recurrent neural network (RNN) cell
    designed for spatiotemporal data. It combines the LSTM's ability to capture long-term dependencies
    with convolutional layers that are adept at handling spatial information.

    Attributes
    ----------
    input_dim: int
        Number of channels of the input tensor.
    hidden_dim: int
        Number of channels of the hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    padding: (int, int)
        Padding added to the input tensor for the convolution operation.
    bias: bool
        Whether or not to add the bias term in convolution.
    conv: nn.Conv2d
        Convolutional layer to transform the combined input and hidden state.

    Methods
    -------
    forward(input_tensor, cur_state):
        Perform a forward pass of the ConvLSTM cell.
    init_hidden(batch_size, image_size):
        Initialize the hidden states for the ConvLSTM cell.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Convolution layer to transform the combined input and hidden state
        self.conv = nn.Conv2d(
            in_channels = self.input_dim + self.hidden_dim,
            out_channels = 4 * self.hidden_dim,
            kernel_size = self.kernel_size,
            padding = self.padding,
            bias = self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch, channels, height, width).
        cur_state: tuple
            Current state containing the hidden state and cell state (h_cur, c_cur).

        Returns
        -------
        h_next: torch.Tensor
            Next hidden state.
        c_next: torch.Tensor
            Next cell state.
        """
        h_cur, c_cur = cur_state

        # Concatenate along the channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolution
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Compute the input, forget, output gates, and cell gate
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Compute the next cell state
        c_next = f * c_cur + i * g
        # Compute the next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden states.

        Parameters
        ----------
        batch_size: int
            Batch size.
        image_size: tuple
            Spatial dimensions of the input (height, width).

        Returns
        -------
        h, c: tuple
            Initialized hidden state and cell state with zeros.
        """
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    Convolutional Long Short-Term Memory (ConvLSTM) module.

    This class implements a multi-layer ConvLSTM, which is designed for spatiotemporal data.
    Each layer of the ConvLSTM is composed of a sequence of ConvLSTM cells, capturing both temporal
    and spatial dependencies in the input data.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index

    Attributes
    ----------
    input_dim: int
        Number of channels in the input tensor.
    hidden_dim: list of int
        Number of hidden channels in each layer.
    kernel_size: list of (int, int)
        Size of the convolutional kernels for each layer.
    num_layers: int
        Number of ConvLSTM layers.
    batch_first: bool
        If True, the input and output tensors are provided as (batch, time, channels, height, width).
    bias: bool
        Whether or not to add the bias term in the convolution.
    return_all_layers: bool
        If True, returns the list of outputs for all layers.

    Methods
    -------
    forward(input_tensor, hidden_state=None):
        Perform a forward pass of the ConvLSTM module.
    _init_hidden(batch_size, image_size):
        Initialize the hidden states for all layers.
    _check_kernel_size_consistency(kernel_size):
        Check the consistency of the kernel size.
    _extend_for_multilayer(param, num_layers):
        Extend parameters for each layer.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        """
        Initialize the ConvLSTM module.

        Parameters
        ----------
        input_dim: int
            Number of channels in the input tensor.
        hidden_dim: int or list of int
            Number of hidden channels in each layer.
        kernel_size: int or list of int
            Size of the convolutional kernels.
        num_layers: int
            Number of ConvLSTM layers.
        batch_first: bool, optional
            If True, the input and output tensors are provided as (batch, time, channels, height, width).
            Default is False.
        bias: bool, optional
            If True, adds a learnable bias to the convolution. Default is True.
        return_all_layers: bool, optional
            If True, returns the list of outputs for all layers. Default is False.
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Ensure that `kernel_size` and `hidden_dim` are lists of length == `num_layers`
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.hidden_states = None

        # Create a list of ConvLSTM cells
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, first_timestep=False, hidden_state=None):
        """
        Forward pass of the ConvLSTM module.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch, time, channels, height, width) if batch_first is True,
            else (time, batch, channels, height, width).
        hidden_state: list of tuples, optional
            Initial hidden state and cell state for each layer. Default is None.

        Returns
        -------
        layer_output_list: list of torch.Tensor
            Output tensor from each layer.
        last_state_list: list of tuples
            Last hidden state and cell state from each layer.
        """
        if not self.batch_first:
            # Convert to (batch, time, channels, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()


        if hidden_state is None or first_timestep or self.hidden_states is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        else:
            hidden_state = self.hidden_states

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # Iterate over each layer
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            # Iterate over each time step
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            # Stack outputs along the time dimension
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        self.hidden_states = last_state_list

        # Optionally return outputs only from the last layer
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden states for all layers.

        Parameters
        ----------
        batch_size: int
            Batch size.
        image_size: tuple
            Spatial dimensions of the input (height, width).

        Returns
        -------
        init_states: list of tuples
            List of initialized hidden state and cell state for each layer.
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check the consistency of kernel_size.

        Parameters
        ----------
        kernel_size: tuple or list of tuples
            Size of the convolutional kernels.

        Raises
        ------
        ValueError
            If kernel_size is not a tuple or a list of tuples.
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extend parameters for each layer.

        Parameters
        ----------
        param: int, list of int
            Parameter to be extended.
        num_layers: int
            Number of layers.

        Returns
        -------
        param: list of int
            Extended parameter for each layer.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class VideoDataset(Dataset):

    def __init__(self, videos, target_videos=None, transform=None, input_target_cut = -1):
        self.videos = videos
        self.target_videos = target_videos
        self.input_target_cut = input_target_cut
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx].values
        if self.transform is not None:
            video = self.apply_transform(video)

        if self.target_videos is not None:
            targ_vid = self.target_videos[idx].values
            if self.transform is not None:
                targ_vid = self.apply_transform(targ_vid)
            return video, targ_vid
        
        # Assume each video has N frames, input and target will be N-1 and 1 respectively
        input_seq = video[:self.input_target_cut]
        target_seq = video[self.input_target_cut:]
        return input_seq, target_seq
    
    def apply_transform(self, video):
        transformed_video = []
        for frame in video:
            transformed_frame = self.transform(frame)
            transformed_video.append(transformed_frame)
        return torch.stack(transformed_video)




class decoder_D(nn.Module):
    def __init__(self, nc=3, nf=64):
        super(decoder_D, self).__init__()
        # Convolution layer to transform the feature maps to the desired number of output channels
        self.d1 = nn.Conv2d(in_channels=nf, out_channels=nc, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, input):
        d1 = self.d1(input)
        return d1

class EncoderRNN(nn.Module):
    def __init__(self, convcell, device, nc=3, nf=64):
        super(EncoderRNN, self).__init__()
        self.decoder_D = decoder_D(nc=nc, nf=nf)
        self.decoder_D = self.decoder_D.to(device)  # Decoder to transform feature maps to output image
        self.convcell = convcell.to(device)  # Convolutional LSTM cell


    def forward(self, input, first_timestep=False):
        output, hidden = self.convcell(input, first_timestep=first_timestep) # Get hidden states and output from ConvLSTM
        output_image = torch.sigmoid(self.decoder_D(output[-1])) # Decode the last output and apply sigmoid
        return output_image


class ConvLSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True):
        super(ConvLSTMEncoderDecoder, self).__init__()
        self.encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=batch_first)
        self.decoder = ConvLSTM(hidden_dim[-1], hidden_dim, kernel_size, num_layers, batch_first=batch_first)

    def forward(self, input_tensor, target_length):
        _, encoder_states = self.encoder(input_tensor)
        decoder_input = input_tensor[:, -1, :, :, :].unsqueeze(1)
        decoder_outputs = []

        for _ in range(target_length):
            decoder_output, encoder_states = self.decoder(decoder_input, encoder_states)
            decoder_input = decoder_output
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs
