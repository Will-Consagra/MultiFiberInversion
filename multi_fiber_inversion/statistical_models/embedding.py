import torch
import torch.nn as nn
import math

"""
spherical U-net CNN code adapted from: https://github.com/AxelElaldi/equivariant-spherical-deconvolution

Conv.state_dict() function removes all attributes associated with graph laplacian, as it is large and needs to be saved as a sparse matrix
        this requires strict=False on the load_state_dict() in order to load the modell, but should not effect prediction as the laplacian is only used during training
"""

####Layers#### 
class Conv(torch.nn.Module):
    """Building Block with a Chebyshev Convolution.
    """

    def __init__(self, in_channels, out_channels, lap, kernel_size=3, bias=True):
        """Initialization.
        Args:
            in_channels (int): initial number of channels
            out_channels (int): output number of channels
            lap (:obj:`torch.sparse.FloatTensor`): laplacian
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1. Defaults to 3.
            bias (bool): Whether to add a bias term.
        """
        super(Conv, self).__init__()
        self.register_buffer("laplacian", lap)
        self.chebconv = ChebConv(in_channels, out_channels, kernel_size, bias)

    def state_dict(self, *args, **kwargs):
        """! WARNING !
        This function overrides the state dict in order to be able to save the model.
        This can be removed as soon as saving sparse matrices has been added to Pytorch.
        """
        state_dict = super().state_dict(*args, **kwargs)
        del_keys = []
        for key in state_dict:
            if key.endswith("laplacian"):
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def forward(self, x):
        """Forward pass.
        Args:
            x (:obj:`torch.tensor`): input [B x Fin x V]
        Returns:
            :obj:`torch.tensor`: output [B x Fout x V]
        """
        x = self.chebconv(self.laplacian, x)
        return x


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """Initialize the Chebyshev layer.
        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
        """
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = cheb_conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, laplacian, inputs):
        """Forward graph convolution.
        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias[None, :, None]
        return outputs


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    # Get tensor dimensions
    B, Fin, V = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials + 1

    # Transform to Chebyshev basis
    x0 = inputs.permute(2, 1, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    inputs = project_cheb_basis(laplacian, x0, K) # K x V x Fin*B

    # Look at the Chebyshev transforms as feature maps at each vertex
    inputs = inputs.view([K, V, Fin, B])  # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    inputs = inputs.view([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout) # K*Fin x Fout
    inputs = inputs.matmul(weight)  # B*V x Fout
    inputs = inputs.view([B, V, Fout])  # B x V x Fout

    # Get final output tensor
    inputs = inputs.permute(0, 2, 1).contiguous()  # B x Fout x V

    return inputs


def project_cheb_basis(laplacian, x0, K):
    """Project vector x on the Chebyshev basis of order K
    \hat{x}_0 = x
    \hat{x}_1 = Lx
    \hat{x}_k = 2*L\hat{x}_{k-1} - \hat{x}_{k-2}
    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        x0 (:obj:`torch.Tensor`): The initial data being forwarded. [V x D]
        K (:obj:`torch.Tensor`): The order of Chebyshev polynomials + 1.
    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev projection.
    """
    inputs = x0.unsqueeze(0)  # 1 x V x D
    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)  # V x D
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)  # 2 x V x D
        for _ in range(2, K):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)  # _ x V x D
            x0, x1 = x1, x2
    return inputs # K x V x D

####Blocks####
class Block(nn.Module):
    """GCNN Unet block.
    """
    def __init__(self, in_ch, int_ch, out_ch, lap, kernel_size):
        """Initialization.
        Args:
            in_ch (int): Number of input channel
            int_ch (int): Number of intermediate channel
            out_ch (int): Number of output channel
            lap (list): Increasing list of laplacians from smallest to largest resolution
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
        """
        super(Block, self).__init__()
        # Conv 1
        self.conv1 = Conv(in_ch, int_ch, lap, kernel_size)
        self.bn1 = nn.BatchNorm1d(int_ch)
        # Conv 2
        self.conv2 = Conv(int_ch, out_ch, lap, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_ch)
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x F_in_ch x V]
        Returns:
            :obj:`torch.Tensor`: output [B x F_out_ch x V]
        """
        x = self.activation(self.bn1(self.conv1(x))) # B x F_int_ch x V
        x = self.activation(self.bn2(self.conv2(x))) # B x F_out_ch x V
        return x


####Spherical U-Net####
class GraphCNNUnet(nn.Module):
    """GCNN Autoencoder.
    """
    def __init__(self, in_channels, out_channels, nvertices, filter_start, kernel_size, pooling, laps):
        """Initialization.
        Args:
            in_channels (int): Number of input channel, i.e., the number of b-value shells
            out_channels (int): Number of output channel: 
            nvertices: number of vertices for graph convolution
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(GraphCNNUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nvertices = nvertices
        self.encoder = Encoder(in_channels, filter_start, kernel_size, pooling.pooling, laps)
        self.decoder = Decoder(out_channels, filter_start, kernel_size, pooling.unpooling, laps)

    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels x V]
        Returns:
            :obj:`torch.Tensor`: output [B x out_channels x V]
        """
        x, enc_ftrs = self.encoder(x)
        x = self.decoder(x, enc_ftrs)
        return x


class Encoder(nn.Module):
    """GCNN Encoder.
    """
    def __init__(self, in_channels, filter_start, kernel_size, pooling, laps):
        """Initialization.
        Args:
            in_channels (int): Number of input channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Encoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.enc_blocks = [Block(in_channels, filter_start, filter_start, laps[-1], kernel_size)]
        self.enc_blocks += [Block((2**i)*filter_start, (2**(i+1))*filter_start, (2**(i+1))*filter_start, laps[-i-2], kernel_size) for i in range(D-2)]
        self.enc_blocks = nn.ModuleList(self.enc_blocks)
        self.pool = pooling
    
    def forward(self, x):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x in_channels x V]          
        Returns:
            :obj:`torch.Tensor`: output [B x (2**(D-2))*filter_start x V_encoded]
            encoder_features (list): Hierarchical encoding. [B x (2**(i))*filter_start x V_encoded_i] for i in [0,D-2]
        """
        ftrs = []
        for block in self.enc_blocks: # len(self.enc_blocks) = D - 2
            x = block(x) # B x (2**(i))*filter_start x V_encoded_i
            ftrs.append(x) 
            x, _ = self.pool(x) # B x (2**(i))*filter_start x V_encoded_(i+1)
        return x, ftrs


class Decoder(nn.Module):
    """GCNN Decoder.
    """
    def __init__(self, out_channels, filter_start, kernel_size, unpooling, laps):
        """Initialization.
        Args:
            out_channels (int): Number of output channel
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(Decoder, self).__init__()
        D = len(laps)
        assert D > 1 # Otherwise there is no encoding/decoding to perform
        self.dec_blocks = [Block((2**(D-2))*filter_start, (2**(D-1))*filter_start, (2**(D-2))*filter_start, laps[0], kernel_size)]
        self.dec_blocks += [Block((2**(D-i))*filter_start, (2**(D-i-1))*filter_start, (2**(D-i-2))*filter_start, laps[i], kernel_size) for i in range(1, D-1)]
        self.dec_blocks += [Block(2*filter_start, filter_start, filter_start, laps[-1], kernel_size)]
        self.dec_blocks = nn.ModuleList(self.dec_blocks)
        self.head = Conv(filter_start, out_channels, laps[-1], kernel_size)
        self.activation = nn.ReLU()
        self.unpool = unpooling

    def forward(self, x, encoder_features):
        """Forward Pass.
        Args:
            x (:obj:`torch.Tensor`): Input to be forwarded. [B x (2**(D-2))*filter_start x V_encoded_(D-1)]
            encoder_features (list): Hierarchical encoding to be forwarded. [B x (2**(i))*filter_start x V_encoded_i] for i in [0,D-2]
        Returns:
            :obj:`torch.Tensor`: output [B x V x out_channels]
        """
        x = self.dec_blocks[0](x) # B x (2**(D-2))*filter_start x V_encoded_(D-1)
        x = self.unpool(x, None) # B x (2**(D-2))*filter_start x V_encoded_(D-2)
        x = torch.cat([x, encoder_features[-1]], dim=1) # B x 2*(2**(D-2))*filter_start x V_encoded_(D-2)
        for i in range(1, len(self.dec_blocks)-1):
            x = self.dec_blocks[i](x) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-1)
            x = self.unpool(x, None) # B x (2**(D-i-2))*filter_start x V_encoded_(D-i-2)
            x = torch.cat([x, encoder_features[-1-i]], dim=1) # B x 2*(2**(D-i-2))*filter_start x V_encoded_(D-i-2)
        x = self.dec_blocks[-1](x) # B x filter_start x V
        x = self.activation(self.head(x)) # B x out_channels x V
        return x

class SphericalSummaryStatistic(nn.Module):
    """GCNN Autoencoder.
    """
    def __init__(self, in_channels, num_equi, num_invar, nvertices, filter_start, kernel_size, pooling, laps):
        """Initialization.
        Args:
            in_channels (int): Number of input channel, i.e., the number of b-value shells
            out_channels (int): Number of output channel: 
            nvertices: number of vertices for graph convolution
            filter_start (int): Number of feature channel after first convolution. Then, multiplied by 2 after every poolig
            kernel_size (int): Size of the kernel (i.e. Order of the Chebyshev polynomials + 1)
            pooling (:obj:'PoolingClass'): Pooling operator
            laps (list): Increasing list of laplacians from smallest to largest resolution, D: size of the list
        """
        super(SphericalSummaryStatistic, self).__init__()

        self.in_channels = in_channels
        out_channels = num_invar + num_equi
        self.n_equi = num_equi
        self.n_inva = num_invar
        self.out_channels = out_channels
        self.nvertices = nvertices
        self._net = GraphCNNUnet(in_channels, out_channels, nvertices, filter_start, kernel_size, pooling, laps)

    def separate(self, x):
        """Separate equivariant and invariant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. [B x out_channels x V]
        Returns:
            x_equi (:obj:`torch.Tensor`): equivariant part of the deconvolution [B x out_channels_equi x V]
            x_inva (:obj:`torch.Tensor`): invariant part of the deconvolution [B x out_channels_inva]
        """
        if self.n_equi != 0:
            x_equi = x[:, :self.n_equi]
        else:
            x_equi = None
        if self.n_inva != 0:
            x_inva = x[:, self.n_equi:]
            x_inva = torch.max(x_inva, dim=2)[0]
        else:
            x_inva = None
        return x_equi, x_inva

    def forward(self, Ctensor):
        """Forward Pass.
        Args:
            Ctensor (:obj:`torch.Tensor`): input to be forwarded. [B x in_channels]
        Returns:
            :obj:`torch.Tensor`: output [B x out_channels x V]
        """
        Spherical_cnn_features = self._net(Ctensor)
        features_equivariant, features_invariant = self.separate(Spherical_cnn_features)
        return {"model_in":Ctensor, "model_out_equi":features_equivariant, "model_out_inv":features_invariant}








