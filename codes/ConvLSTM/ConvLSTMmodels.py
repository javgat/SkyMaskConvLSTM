"""Module with the definition of ConvLSTM classes

Initial work from: https://github.com/ndrplz/ConvLSTM_pytorch
"""

import random
from typing import List, Tuple, Union

import torch
import torch.nn as nn


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
                 device='cpu', batch_first=False, bias=True, return_all_layers=False,
                 batch_normalization=False):
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
        batch_normalization: bool, optional
            If True, batch normalization and relu activation will be performed between layers.
            Default is False.
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
        self.batch_normalization = batch_normalization

        if self.batch_normalization:
            # Create batch normalization elements
            self.batch_norms = nn.ModuleList([nn.BatchNorm3d(h, device=device) for h in hidden_dim[:-1]])
            self.activations = nn.ModuleList([nn.ReLU() for _ in hidden_dim[:-1]])

        # Create a list of ConvLSTM cells
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, first_timestep=False):
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


        if first_timestep or (hidden_state is None and self.hidden_states is None):
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        elif hidden_state is None and self.hidden_states is not None:
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

            if self.batch_normalization and layer_idx != self.num_layers-1:
                layer_output = self.batch_norms[layer_idx](layer_output.permute((0, 2, 1, 3, 4)))
                layer_output = self.activations[layer_idx](layer_output).permute((0, 2, 1, 3, 4))

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


class RNNConvLSTM(nn.Module):
    def __init__(self, hidden_dims: list, kernel_size: Union[List[Tuple[int]], Tuple[int]], num_layers: int=1, device='cpu', n_channels: int = 3):
        super(RNNConvLSTM, self).__init__()
        self.n_channels = n_channels # 3
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.enc = ConvLSTM(n_channels, hidden_dims, kernel_size, num_layers, device, batch_first=True, return_all_layers=True).to(device)
        self.fc = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=n_channels, kernel_size=(1, 1), stride=1, padding=0).to(device)


    def forward(self, video, target_len: int,  target_seq=None, teacher_forcing_ratio=0.5, return_source_prediction=False):
        # b, t, c, i0, i1
        video = video.to(self.device)
        if target_seq is not None:
            target_seq = target_seq.to(self.device)
        ## LSTM
        batch_size = video.size(0)
        source_len = video.size(1)
        target_size = video.size(2)
        hidden_lstm_s2s = None
        if return_source_prediction:
            lstm_out_source = torch.zeros((batch_size, source_len-1, target_size, video.size(3), video.size(4))).to(video.device)
        for t in range(source_len-1):
            lstm_dec_input = video[:, t].unsqueeze(1)
            if hidden_lstm_s2s is not None:
                out, hidden_lstm_s2s = self.enc(lstm_dec_input, hidden_lstm_s2s)
            else:
                out, hidden_lstm_s2s = self.enc(lstm_dec_input, first_timestep=True)
            if return_source_prediction:
                out = out[-1].flatten(0, 1)
                out = self.fc(out).unsqueeze(1)
                lstm_out_source[:, t, :] = out.squeeze(1)

        lstm_dec_input = video[:, -1].unsqueeze(1)
        lstm_out = torch.zeros((batch_size, target_len, target_size, video.size(3), video.size(4))).to(video.device)
        for t in range(target_len):
            out, hidden_lstm_s2s = self.enc(lstm_dec_input, hidden_lstm_s2s)
            out = out[-1].flatten(0, 1)
            out = self.fc(out).unsqueeze(1)
            lstm_out[:, t, :] = out.squeeze(1)
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                lstm_dec_input = target_seq[:, t].unsqueeze(1)
            else:
                lstm_dec_input = out

        lstm_out = lstm_out.flatten(0,1)
        ##
        lstm_out = torch.unflatten(lstm_out, 0, (batch_size, target_len))

        if return_source_prediction:
            lstm_out_source = lstm_out_source.flatten(0,1)
            lstm_out_source = torch.unflatten(lstm_out_source, 0, (batch_size, source_len-1))
            lstm_out = torch.cat([lstm_out_source, lstm_out], 1)
            
        return lstm_out


class SegmentedConvLSTMNet(nn.Module):
    def __init__(self, hidden_dims: List[int], kernel_size: Union[List[Tuple[int]], Tuple[int]], num_layers: int=1, device='cpu', n_channels: int = 5, batch_normalization: bool = True):
        super(SegmentedConvLSTMNet, self).__init__()
        self.n_channels = n_channels
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.batch_normalization = batch_normalization

        # Define ConvLSTM net
        self.enc = ConvLSTM(n_channels, hidden_dims, kernel_size, num_layers, device, batch_first=True, return_all_layers=True, batch_normalization=batch_normalization).to(device)
        
        # Batch Normalization and ReLU activation
        ## BN2d because ConvLSTM is used for outputing frame by frame data
        if self.batch_normalization:
            self.batch_norm = nn.BatchNorm2d(hidden_dims[-1], device=device)
            self.activation = nn.ReLU()

        # Output layer
        ## Conv2d because ConvLSTM outputs frames one by one
        self.fc = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=n_channels, kernel_size=(1, 1), stride=1, padding=0).to(device)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, video, target_len: int, target_seq=None, teacher_forcing_ratio=0.5, return_source_prediction=False):
        # b, t, c, i0, i1
        video = video.to(self.device)
        if target_seq is not None:
            target_seq = target_seq.to(self.device)
        ## LSTM
        batch_size = video.size(0)
        source_len = video.size(1)
        hidden_lstm_s2s = None

        if return_source_prediction:
            lstm_out_source = torch.zeros((batch_size, source_len-1, self.n_channels, video.size(3), video.size(4))).to(video.device)
        for t in range(source_len - 1):
            lstm_dec_input = video[:, t].unsqueeze(1)
            if hidden_lstm_s2s is not None:
                out, hidden_lstm_s2s = self.enc(lstm_dec_input, hidden_lstm_s2s)
            else:
                out, hidden_lstm_s2s = self.enc(lstm_dec_input, first_timestep=True)
            if return_source_prediction:
                out = out[-1].flatten(0, 1)
                if self.batch_normalization:
                    out = self.batch_norm(out)
                    out = self.activation(out)
                out = self.fc(out)
                out = self.softmax(out).unsqueeze(1)
                lstm_out_source[:, t, :] = out.squeeze(1)

        lstm_dec_input = video[:, -1].unsqueeze(1)
        lstm_out = torch.zeros((batch_size, target_len, self.n_channels, video.size(3), video.size(4))).to(video.device)

        for t in range(target_len):
            out, hidden_lstm_s2s = self.enc(lstm_dec_input, hidden_lstm_s2s)
            out = out[-1].flatten(0, 1)
            if self.batch_normalization:
                out = self.batch_norm(out)
                out = self.activation(out)
            out = self.fc(out)
            out = self.softmax(out).unsqueeze(1)
            lstm_out[:, t, :] = out.squeeze(1)
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                lstm_dec_input = target_seq[:, t].unsqueeze(1)
            else:
                lstm_dec_input = out

        lstm_out = lstm_out.flatten(0, 1)
        lstm_out = torch.unflatten(lstm_out, 0, (batch_size, target_len))

        if return_source_prediction:
            lstm_out_source = lstm_out_source.flatten(0, 1)
            lstm_out_source = torch.unflatten(lstm_out_source, 0, (batch_size, source_len - 1))
            lstm_out = torch.cat([lstm_out_source, lstm_out], 1)
        
        return lstm_out


class Seq2SeqConvLSTM(nn.Module):
    def __init__(self, hidden_dims: list, kernel_size: Union[List[Tuple[int]], Tuple[int]], num_layers: int=1, device='cpu', n_channels: int = 3):
        super(Seq2SeqConvLSTM, self).__init__()
        self.n_channels = n_channels # 3
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        # ConvLSTM as Seq2Seq LSTM model to customize output length. (?)
        self.enc = ConvLSTM(n_channels, hidden_dims, kernel_size, num_layers, device, batch_first=True, return_all_layers=True).to(device)
        self.dec = ConvLSTM(n_channels, hidden_dims, kernel_size, num_layers, device, batch_first=True, return_all_layers=True).to(device)
        self.fc = nn.Conv2d(in_channels=hidden_dims[-1], out_channels=n_channels, kernel_size=(1, 1), stride=1, padding=0).to(device)


    def forward(self, video, target_len: int,  target_seq=None, teacher_forcing_ratio=0.5):
        # b, t, c, i0, i1
        video = video.to(self.device)
        if target_seq is not None:
            target_seq = target_seq.to(self.device)
        seq_dims = video.shape[0:2]
        ## LSTM
        _, hidden_lstm_s2s = self.enc(video, first_timestep=True)
        batch_size = video.size(0)
        target_size = video.size(2)
        lstm_dec_input = video[:, -1].unsqueeze(1)
        lstm_out = torch.zeros((batch_size, target_len, target_size, video.size(3), video.size(4))).to(video.device)
        for t in range(target_len):
            out, hidden_lstm_s2s = self.dec(lstm_dec_input, hidden_lstm_s2s)
            out = out[-1].flatten(0, 1)
            out = self.fc(out).unsqueeze(1)
            lstm_out[:, t, :] = out.squeeze(1)
            if target_seq is not None and random.random() < teacher_forcing_ratio:
                lstm_dec_input = target_seq[:, t].unsqueeze(1)
            else:
                lstm_dec_input = out

        lstm_out = lstm_out.flatten(0,1)
        ##
        lstm_out = torch.unflatten(lstm_out, 0, (seq_dims[0], target_len))
        return lstm_out
