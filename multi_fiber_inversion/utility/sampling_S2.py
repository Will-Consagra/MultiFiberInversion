import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import sparse
from scipy import special as sci
import math

from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from scipy.sparse import coo_matrix

"""
Code adapted from: https://github.com/AxelElaldi/equivariant-spherical-deconvolution
"""

def prepare_laplacian(laplacian):
    """Prepare a graph Laplacian to be fed to a graph convolutional layer.
    Args:
        laplacian :obj:'scipy.sparse.csr.csr_matrix': sparse numpy laplacian
    Returns:
        :obj:`torch.sparse.FloatTensor: Scaled, shifted and sparse torch laplacian
    """

    def estimate_lmax(laplacian, tol=5e-3):
        """Estimate the largest eigenvalue of an operator.
        """
        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol, ncv=min(laplacian.shape[0], 10), return_eigenvectors=False)
        lmax = lmax[0]
        lmax *= 1 + 2 * tol  # Be robust to errors.
        return lmax

    def scale_operator(L, lmax, scale=1):
        """Scale the eigenvalues from [0, lmax] to [-scale, scale].
        """
        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    lmax = estimate_lmax(laplacian)
    laplacian = scale_operator(laplacian, lmax)
    laplacian = scipy_csr_to_sparse_tensor(laplacian)
    return laplacian


def scipy_csr_to_sparse_tensor(csr_mat):
    """Convert scipy csr to sparse pytorch tensor.
    Args:
        csr_mat :obj:'scipy.sparse.csr.csr_matrix': The sparse scipy matrix.
    Returns:
        sparse_tensor :obj:`torch.sparse.FloatTensor`: The sparse torch matrix.
    """
    coo = coo_matrix(csr_mat)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    idx = torch.LongTensor(indices)
    vals = torch.FloatTensor(values)
    shape = coo.shape
    sparse_tensor = torch.sparse.FloatTensor(idx, vals, torch.Size(shape))
    sparse_tensor = sparse_tensor.coalesce()
    return sparse_tensor


def healpix_resolution_calculator(nodes):
    """Calculate the resolution of a healpix graph
    for a given number of nodes.
    Args:
        nodes (int): number of nodes in healpix sampling
    Returns:
        int: resolution for the matching healpix graph
    """
    resolution = int(math.sqrt(nodes / 12))
    return resolution

class Healpix:
    """Healpix class, which groups together the corresponding pooling and unpooling.
    """

    def __init__(self, mode="average"):
        """Initialize healpix pooling and unpooling objects.
        Args:
            mode (str, optional): specify the mode for pooling/unpooling.
                                    Can be maxpooling or averagepooling. Defaults to 'average'.
        """
        if mode == "max":
            self.__pooling = HealpixMaxPool()
            self.__unpooling = HealpixMaxUnpool()
        else:
            self.__pooling = HealpixAvgPool()
            self.__unpooling = HealpixAvgUnpool()

    @property
    def pooling(self):
        """Get pooling
        """
        return self.__pooling

    @property
    def unpooling(self):
        """Get unpooling
        """
        return self.__unpooling

## Max Pooling/Unpooling
class HealpixMaxPool(nn.MaxPool1d):
    """Healpix Maxpooling module
    """

    def __init__(self):
        """Initialization
        """
        super().__init__(kernel_size=4, return_indices=True)

    def forward(self, x):
        """Forward call the 1d Maxpooling of pytorch
        Args:
            x (:obj:`torch.tensor`):[B x Fin x V]
        Returns:
            tuple((:obj:`torch.tensor`), indices (int)): [B x Fin x V_pool] and indices of pooled pixels
        """
        x, indices = F.max_pool1d(x, self.kernel_size, return_indices=True) # B x Fin x V_pool
        return x, indices


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """Healpix Maxunpooling using the MaxUnpool1d of pytorch
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x, indices):
        """calls MaxUnpool1d using the indices returned previously by HealpixMaxPool
        Args:
            tuple(x (:obj:`torch.tensor`) : [B x Fin x V]
            indices (int)): indices of pixels equiangular maxpooled previously
        Returns:
            [:obj:`torch.tensor`] -- [B x Fin x V_unpool]
        """
        x = F.max_unpool1d(x, indices, self.kernel_size) # B x Fin x V_unpool
        return x


## Avereage Pooling/Unpooling
class HealpixAvgPool(nn.AvgPool1d):
    """Healpix Average pooling module
    """

    def __init__(self):
        """initialization
        """
        super().__init__(kernel_size=4)

    def forward(self, x):
        """forward call the 1d Averagepooling of pytorch
        Arguments:
            x (:obj:`torch.tensor`): [B x Fin x V]
        Returns:
            tuple((:obj:`torch.tensor`), indices (None)): [B x Fin x V_pool] and indices for consistence
            with maxPool
        """
        x = F.avg_pool1d(x, self.kernel_size) # B x Fin x V_pool
        return x, None


class HealpixAvgUnpool(nn.Module):
    """Healpix Average Unpooling module
    """

    def __init__(self):
        """initialization
        """
        self.kernel_size = 4
        super().__init__()

    def forward(self, x, indices):
        """forward repeats (here more like a numpy tile for the moment) the incoming tensor
        Arguments:
            x (:obj:`torch.tensor`), indices (None): [B x Fin x V] and indices for consistence with maxUnPool
        Returns:
            [:obj:`torch.tensor`]: [B x Fin x V_unpool]
        """
        x = F.interpolate(x, scale_factor=self.kernel_size, mode="nearest") # B x Fin x V_unpool
        return x

class HealpixSampling:
    """Graph Spherical sampling class.
    """
    def __init__(self, n_side, depth, sh_degree=None, pooling_mode='average'):
        """Initialize the sampling class.
        Args:
            n_side (int): Healpix resolution
            depth (int): Depth of the encoder
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            pooling_mode (str, optional): specify the mode for pooling/unpooling.
                                            Can be max or average. Defaults to 'average'.
        """
        assert math.log(n_side, 2).is_integer()
        assert n_side / (2**(depth-1)) >= 1

        G = SphereHealpix(n_side, nest=True, k=8) # Highest resolution sampling
        self.sampling = Sampling(G.coords, sh_degree)
        print(self.sampling.S2SH.shape[1], (sh_degree+1)*(sh_degree//2+1))
        assert self.sampling.S2SH.shape[1] == (sh_degree+1)*(sh_degree//2+1)
        
        self.laps = self.get_healpix_laplacians(n_side, depth, laplacian_type="normalized", neighbor=8)
        self.pooling = Healpix(mode=pooling_mode)
    
    def get_healpix_laplacians(self, starting_nside, depth, laplacian_type, neighbor=8):
        """Get the healpix laplacian list for a certain depth.
        Args:
            starting_nside (int): initial healpix grid resolution.
            depth (int): the depth of the UNet.
            laplacian_type ["combinatorial", "normalized"]: the type of the laplacian.
        Returns:
            laps (list): increasing list of laplacians from smallest to largest resolution
        """
        laps = []
        for i in range(depth):
            n_side = starting_nside//(2**i) # Get resolution of the grid at depth i
            G = SphereHealpix(n_side, nest=True, k=neighbor) # Construct Healpix Graph at resolution n_side
            G.compute_laplacian(laplacian_type) # Compute Healpix laplacian
            laplacian = prepare_laplacian(G.L) # Get Healpix laplacian
            laps.append(laplacian)
        return laps[::-1]

class Sampling:
    """Spherical sampling class.
    """

    def __init__(self, vectors, sh_degree=None, max_sh_degree=None, constant=False):
        """Initialize symmetric sampling class.
        Args:
            vectors (np.array): [V x 3] Sampling position on the unit sphere (bvecs)
            sh_degree (int, optional): Spherical harmonic degree of the sampling
            max_sh_degree (int, optional): Max Spherical harmonic degree of the sampling if sh_degree is None
            constant (bool, optional): In the case of a shell==0
        """
        # Load sampling
        assert vectors.shape[1] == 3
        self.vectors = vectors # V x 3

        # Compute sh_degree
        if sh_degree is None:
            sh_degree = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4) # We want the number of SHC to be at most the number of vectors
            if not max_sh_degree is None:
                sh_degree = min(sh_degree, max_sh_degree)
        if constant:
            self.S2SH = np.ones((vectors.shape[0], 1)) * math.sqrt(4*math.pi) # V x 1
            self.SH2S = np.zeros(((sh_degree+1)*(sh_degree//2+1), vectors.shape[0])) # (sh_degree+1)(sh_degree//2+1) x V 
            self.SH2S[0] = 1 / math.sqrt(4*math.pi)
        else:
            # Compute SH matrices
            _, self.SH2S = self.sh_matrix(sh_degree, vectors, with_order=1) # (sh_degree+1)(sh_degree//2+1) x V 
            
            # We can't recover more SHC than the number of vertices:
            sh_degree_s2sh = 2*int((np.sqrt(8*vectors.shape[0]-7) - 3) / 4)
            sh_degree_s2sh = min(sh_degree_s2sh, sh_degree)
            if not max_sh_degree is None:
                sh_degree_s2sh = min(sh_degree_s2sh, max_sh_degree)
            self.S2SH, _ = self.sh_matrix(sh_degree_s2sh, vectors, with_order=1) # V x (sh_degree_s2sh+1)(sh_degree_s2sh//2+1)

    def sh_matrix(self, sh_degree, vectors, with_order):
        return _sh_matrix(sh_degree, vectors, with_order)


def _sh_matrix(sh_degree, vector, with_order=1):
    """
    Create the matrices to transform the signal into and from the SH coefficients.

    A spherical signal S can be expressed in the SH basis:
    S(theta, phi) = SUM c_{i,j} Y_{i,j}(theta, phi)
    where theta, phi are the spherical coordinates of a point
    c_{i,j} is the spherical harmonic coefficient of the spherical harmonic Y_{i,j}
    Y_{i,j} is the spherical harmonic of order i and degree j

    We want to find the coefficients c from N known observation on the sphere:
    S = [S(theta_1, phi_1), ... , S(theta_N, phi_N)]

    For this, we use the matrix
    Y = [[Y_{0,0}(theta_1, phi_1)             , ..., Y_{0,0}(theta_N, phi_N)                ],
        ................................................................................... ,
        [Y_{sh_order,sh_order}(theta_1, phi_1), ... , Y_{sh_order,sh_order}(theta_N, phi_N)]]

    And:
    C = [c_{0,0}, ... , c_{sh_order,sh_order}}

    We can express S in the SH basis:
    S = C*Y


    Thus, if we know the signal SH coefficients C, we can find S with:
    S = C*Y --> This code creates the matrix Y

    If we known the signal Y, we can find C with:
    C = S * Y^T * (Y * Y^T)^-1  --> This code creates the matrix Y^T * (Y * Y^T)^-1

    Parameters
    ----------
    sh_degree : int
        Maximum spherical harmonic degree
    vector : np.array (N_grid x 3)
        Vertices of the grid
    with_order : int
        Compute with (1) or without order (0)
    Returns
    -------
    spatial2spectral : np.array (N_grid x N_coef)
        Matrix to go from the spatial signal to the spectral signal
    spectral2spatial : np.array (N_coef x N_grid)
        Matrix to go from the spectral signal to the spatial signal
    """
    if with_order not in [0, 1]:
        raise ValueError('with_order must be 0 or 1, got: {0}'.format(with_order))

    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    colats = np.arccos(z)
    lons = np.arctan2(y, x) % (2 * np.pi)
    grid = (colats, lons)
    gradients = np.array([grid[0].flatten(), grid[1].flatten()]).T

    num_gradients = gradients.shape[0]
    if with_order == 1:
        num_coefficients = int((sh_degree + 1) * (sh_degree/2 + 1))
    else:
        num_coefficients = sh_degree//2 + 1

    b = np.zeros((num_coefficients, num_gradients))
    for id_gradient in range(num_gradients):
        id_column = 0
        for id_degree in range(0, sh_degree + 1, 2):
            for id_order in range(-id_degree * with_order, id_degree * with_order + 1):
                gradients_phi, gradients_theta = gradients[id_gradient]
                y = sci.sph_harm(np.abs(id_order), id_degree, gradients_theta, gradients_phi)
                if id_order < 0:
                    b[id_column, id_gradient] = np.imag(y) * np.sqrt(2)
                elif id_order == 0:
                    b[id_column, id_gradient] = np.real(y)
                elif id_order > 0:
                    b[id_column, id_gradient] = np.real(y) * np.sqrt(2)
                id_column += 1

    b_inv = np.linalg.inv(np.matmul(b, b.transpose()))
    spatial2spectral = np.matmul(b.transpose(), b_inv)
    spectral2spatial = b
    return spatial2spectral, spectral2spatial

