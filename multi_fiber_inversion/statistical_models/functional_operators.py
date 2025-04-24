import torch 
import torch.nn as nn
import numpy as np
import scipy.special as sps

def cart2sphere(xtensor):
    ## Note: theta, phi convention here is flipped compared to dipy.core.geometry.cart2sphere
    x = xtensor.cpu().detach().numpy()
    r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
    theta = np.arctan2(x[:,1], x[:,0])
    phi = np.arccos(x[:,2]/r)
    return torch.from_numpy(np.column_stack([theta, phi])).float()

def matern_spec_density(omega, rho, nu):
    """
    Spectral density for a Matern covariance function. Form can be found in Dutordoir et. al 2020 supplement.
    arguments:
        omega: frequency 
        rho: lengthscale 
        nu: differentiability
    """
    term1 = ((2**3) * (np.pi**(3/2)) * sps.gamma(nu + (3/2)) * np.power(2*nu, nu))/(sps.gamma(nu) * np.power(rho, 2*nu))
    term2 = np.power(((2*nu)/np.power(rho, 2)) + (4*(np.pi**2)*np.power(omega,2)), -(nu + (3/2)))
    return term1 * term2

def sph_harm_ind_list(sh_order, full_basis=False):
    """
    Returns the degree (``m``) and order (``n``) of all the symmetric spherical
    harmonics of degree less then or equal to ``sh_order``. The results,
    ``m_list`` and ``n_list`` are kx1 arrays, where k depends on ``sh_order``.
    They can be passed to :func:`real_sh_descoteaux_from_index` and
    :func:``real_sh_tournier_from_index``.

    Parameters
    ----------
    sh_order : int
        even int > 0, max order to return
    full_basis: bool, optional
        True for SH basis with even and odd order terms

    Returns
    -------
    m_list : array
        degrees of even spherical harmonics
    n_list : array
        orders of even spherical harmonics

    See Also
    --------
    shm.real_sh_descoteaux_from_index, shm.real_sh_tournier_from_index

    """
    if full_basis:
        n_range = np.arange(0, sh_order + 1, dtype=int)
        ncoef = int((sh_order + 1) * (sh_order + 1))
    else:
        if sh_order % 2 != 0:
            raise ValueError('sh_order must be an even integer >= 0')
        n_range = np.arange(0, sh_order + 1, 2, dtype=int)
        ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
    n_list = np.repeat(n_range, n_range * 2 + 1)
    offset = 0
    m_list = np.empty(ncoef, 'int')
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = np.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1
    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return m_list, n_list

def spherical_harmonics(m, n, theta, phi, use_scipy=True):
    """Compute spherical harmonics.

    This may take scalar or array arguments. The inputs will be broadcast
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The degree of the harmonic.
    n : int ``>= 0``
        The order of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.
    use_scipy : bool, optional
        If True, use scipy implementation.

    Returns
    -------
    y_mn : complex float
        The harmonic $Y^m_n$ sampled at ``theta`` and ``phi``.

    Notes
    -----
    This is a faster implementation of scipy.special.sph_harm for
    scipy version < 0.15.0. For scipy 0.15 and onwards, we use the scipy
    implementation of the function.

    The usual definitions for ``theta` and `phi`` used in DIPY are interchanged
    in the method definition to agree with the definitions in
    scipy.special.sph_harm, where `theta` represents the azimuthal coordinate
    and `phi` represents the polar coordinate.

    Although scipy uses a naming convention where ``m`` is the order and ``n``
    is the degree of the SH, the opposite of DIPY's, their definition for both
    parameters is the same as ours, with ``n >= 0`` and ``|m| <= n``.
    """
    if use_scipy:
        return sps.sph_harm(m, n, theta, phi, dtype=complex)
    x = np.cos(phi)
    val = sps.lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (sps.gammaln(n - m + 1) - sps.gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val

def real_sym_sh_basis(sh_order, theta, phi, full_basis=False, legacy=True):
    m, n = sph_harm_ind_list(sh_order, full_basis)

    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])
    if legacy:
        sh = spherical_harmonics(np.abs(m), n, phi, theta)
    else:
        sh = spherical_harmonics(m, n, phi, theta)

    real_sh = np.where(m > 0, sh.imag, sh.real)
    real_sh *= np.where(m == 0, 1., np.sqrt(2))

    return real_sh

class HarmonicRegression(torch.nn.Module):
    def __init__(self, device, design, sh_order=8):
        """Initialization.
        Args:
            design (:obj:`torch.Tensor`): S2-grid to sample signal grid scheme (nverts x 3) (save map for standard grid so we don't need to re-compute inverse each sim call)
            sh_order int: max order of harmonic basis
        """
        super().__init__()
        self.sh_order = sh_order
        self.K = int((sh_order+1)*(sh_order+2)/2)
        m, n = sph_harm_ind_list(self.sh_order, False)
        self.m = m
        self.n = n
        X_sph = cart2sphere(design).detach().cpu().numpy()
        theta_x = X_sph[:,0]; phi_x = X_sph[:,1]
        Bmat = real_sym_sh_basis(sh_order, phi_x, theta_x)
        self.S2CMat = torch.from_numpy(np.linalg.inv(Bmat.T @  Bmat) @ Bmat.T).float().to(device) ##K x nverts
        self.C2SMat = torch.from_numpy(Bmat).float().to(device) ## nverts x K
    def harmonic_expansion(self, X):
        """    
        Args:
            X (:obj:`torch.Tensor`): nbatch x 3  
        Returns:
            Phi_evals :obj:`torch.Tensor`: harmonic basis evaluations: nbatch x K
        """
        X_sph = cart2sphere(X).detach().cpu().numpy()
        theta_x = X_sph[:,0]; phi_x = X_sph[:,1]
        Phi_evals = real_sym_sh_basis(self.sh_order, phi_x, theta_x)
        return torch.from_numpy(Phi_evals).float()
    def S2C(self, Stensor):
        """(Linear S to C mapping)
        Args:
            Stensor (:obj:`torch.Tensor`): nbatch x nchannels x nverts  
        Returns:
            Ctensor :obj:`torch.Tensor`: forward model evalutations on dense grid, nbatch x nchannels x K
        """
        Ctensor = Stensor @ self.S2CMat.T 
        return Ctensor
    def forward(self, Ctensor):
        """Forward Pass (Linear C2S mapping)
        Args:
            Ctensor (:obj:`torch.Tensor`): nbatch x nchannels x K  
        Returns:
            Stensor :obj:`torch.Tensor`: forward model evalutations on dense grid, nbatch x nchannels x nverts
        """
        Stensor = Ctensor @ self.C2SMat.T
        return Stensor


class OperatorNetwork(nn.Module):
    """
    Operator G(C) = log (g), were C \in H_1 \otimes \cdots H_{L} (L is the number of b-values), and log(g) \in {v: v\in H and \int \exp(v) = 1}
    ----------
    FuncSpace: nn.Module, function space to represent posterior fodf 
    summary_network: spherical CNN definining coefficient functions.
    Q_grid: grid to compute quadrature  
    """
    def __init__(self, FuncSpace, summary_network, quad_points, quad_weights):

        super().__init__()

        self._function_space = FuncSpace
        self._coordinate_dims = FuncSpace.K

        self._hidden_net = summary_network
        hidden_features = summary_network.out_channels*summary_network.nvertices
        self._hidden_features  = hidden_features
        
        self._final_layer = nn.Linear(hidden_features, self._coordinate_dims) 

        quad_sph = cart2sphere(quad_points).detach().cpu().numpy()
        theta_q = quad_sph[:,0]; phi_q = quad_sph[:,1]
        self.Phi_quad = torch.from_numpy(real_sym_sh_basis(self._function_space.sh_order, phi_q, phi_q)).float()
        self.quad_weights = quad_weights 

    def forward(self,  context):
        """
        context: Tensor of observed signal functions 
        """
        hidden_net_results = self._hidden_net(context)
        h = hidden_net_results["model_out_equi"] ##nbatch X n_equi X self._hidden_net.nvertices
        h = h.view(-1, self._hidden_features)
        ## compute unconstrained density function (should be rotationally equivariant)
        log_v_coefs = self._final_layer(h)
        ## approximate normalization integral
        log_v_norm = torch.log(torch.sum(torch.exp(log_v_coefs @ self.Phi_quad.T) * self.quad_weights, axis=1))
        return log_v_coefs, log_v_norm 

####Kernel Functions for GPs####

class SphericalMaternKernel(torch.nn.Module):
    def __init__(self, harmonic_space, rho, nu):
        super().__init__()
        self.rho = rho 
        self.nu = nu 
        self.harmonic_space = harmonic_space
        self.SpecMat = torch.from_numpy(np.diag(matern_spec_density(np.sqrt(harmonic_space.n*(harmonic_space.n+1)), rho, nu))).float()
        #self.SpecMat[0,0] = torch.tensor(0) ##force mean-zero

    def forward(self, X1, X2):
        Phi_X1 = self.harmonic_space.harmonic_expansion(X1) ##nbatch1 x K 
        Phi_X2 = self.harmonic_space.harmonic_expansion(X2) ##nbatch2 x K 
        return Phi_X1[:,1:] @ self.SpecMat[1:,1:] @ Phi_X2[:,1:].T##nbatch1 x nbatch2

class RadialSquaredExpKernel(torch.nn.Module):
    def __init__(self, sigma2_l):
        super().__init__()
        self.prec_l = 1./sigma2_l

    def forward(self, b1, b2): 
        return torch.exp(-0.5*self.prec_l*torch.pow(torch.cdist(torch.log(b1), torch.log(b2), p=2), 2)) ##nbatch1 x nbatch2


