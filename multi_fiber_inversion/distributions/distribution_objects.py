import numpy as np
import torch
from torch import Tensor, float32
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.uniform import Uniform
from torch.distributions import constraints, transforms
from torch.distributions.dirichlet import Dirichlet
from typing import Generator, Iterable, List, Optional, Tuple

def cart2sphere(xtensor):
	## Note: theta, phi convention here is flipped compared to dipy.core.geometry.cart2sphere
	x = xtensor.detach().numpy()
	r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
	theta = np.arctan2(x[:,1], x[:,0])
	phi = np.arccos(x[:,2]/r)
	return torch.from_numpy(np.column_stack([theta, phi])).float()

def sphere2cart(xtensor):
	x = xtensor.detach().numpy()
	theta = x[:,0]
	phi = x[:,1]
	xx = np.sin(phi)*np.cos(theta)
	yy = np.sin(phi)*np.sin(theta)
	zz = np.cos(phi)
	return torch.from_numpy(np.column_stack([xx, yy, zz])).float()

def S2hemisphereIndicator(x):
	x_copy = x.clone() 
	x_polar = cart2sphere(x_copy)
	ix = (x_polar[:,1] < np.pi/2)
	return ix

class BoxUniform(Independent):
    def __init__(self, low, high, reinterpreted_batch_ndims: int = 1, device: str = "cpu"):
        """Multidimensional uniform distribution defined on a box.
        Args:
            low: lower range (inclusive).
            high: upper range (exclusive).
            reinterpreted_batch_ndims (int): the number of batch dims to
                                             reinterpret as event dims.
            device: device of the prior, defaults to "cpu", should match the training
                device when used in SBI.
        """
        super().__init__(
            Uniform(
                low=torch.as_tensor(
                    low, dtype=torch.float32, device=torch.device(device)
                ),
                high=torch.as_tensor(
                    high, dtype=torch.float32, device=torch.device(device)
                ),
                validate_args=False,
            ),
            reinterpreted_batch_ndims,
        )

class UniformAxial(Distribution): 
	"""
	Uniform distribution over axial data on S^2 
	"""
	def __init__(self):
		batch_shape = torch.Size()
		super().__init__(batch_shape, validate_args=False)

	def rsample(self, sample_shape=torch.Size(),
					 seed: Optional[int] = None) -> Tensor:
		shape = self._extended_shape(sample_shape)
		nsamples = shape.numel()
		samples = torch.zeros((nsamples,3), dtype=torch.float32)
		done = torch.zeros((nsamples,), dtype=torch.bool)
		while not done.all():
			rnormals = torch.normal(0, 1, size=(nsamples,3), dtype=torch.float32)
			candidates = rnormals/torch.norm(rnormals, dim = 1, p = 2)[:,None]
			fsb = S2hemisphereIndicator(candidates)
			if fsb.any():
				samples[fsb,:] = candidates[fsb,:]
				done = done | fsb
		return samples

	def log_prob(self, theta: Tensor) -> Tensor:
		return torch.log(4*np.pi*S2hemisphereIndicator(theta).float())

class ConvexCombination(Distribution):
	"""
	Samples over the n-dimensional simplex with min weight parameter
	"""
	def __init__(self, num_mix=2, min_weight=0.2, ordered=True):
		self.min_weight = min_weight 
		self.num_mix = num_mix
		self.ordered = ordered
		self.dir_dist = Dirichlet(torch.tensor([1./self.num_mix]*self.num_mix))
		batch_shape = torch.Size()
		super().__init__(batch_shape, validate_args=False)

	def rsample(self, sample_shape=torch.Size(),
					 seed: Optional[int] = None) -> Tensor:
		shape = self._extended_shape(sample_shape)
		nsamples = shape.numel()
		samples = torch.zeros((nsamples,self.num_mix), dtype=torch.float32)
		done = torch.zeros((nsamples,), dtype=torch.bool)
		while not done.all():
			candidates = self.dir_dist.sample((nsamples,))
			fsb = torch.all(candidates > self.min_weight, axis=1)
			if fsb.any():
				samples[fsb,:] = candidates[fsb,:]
				done = done | fsb
		if self.ordered:
			samples, indices = torch.sort(samples, descending=True, dim=1) 
		return samples

	def log_prob(self, theta: Tensor) -> Tensor:
		return self.dir_dist.log_prob(theta)

class ConvexCombinationBiHierarchical(Distribution):
	"""
	Samples over the n-dimensional simplex with min weight parameter
	"""
	def __init__(self, num_mix=2, num_comp=2, min_fib_w=0.2, min_comp_w=0.05):
		self.min_fib_w = min_fib_w 
		self.min_comp_w = min_comp_w
		self.num_mix = num_mix
		self.num_comp = num_comp
		self.dir_dist = Dirichlet(torch.tensor([1./(self.num_comp*self.num_mix)]*(self.num_comp*self.num_mix)))
		batch_shape = torch.Size()
		super().__init__(batch_shape, validate_args=False)

	def rsample(self, sample_shape=torch.Size(),
					 seed: Optional[int] = None) -> Tensor:
		shape = self._extended_shape(sample_shape)
		nsamples = shape.numel()
		samples = torch.zeros((nsamples, self.num_mix, self.num_comp), dtype=torch.float32)
		done = torch.zeros((nsamples,), dtype=torch.bool)
		while not done.all():
			candidates = self.dir_dist.sample((nsamples,)).view(nsamples, self.num_mix, self.num_comp)
			fsb_1 = torch.all(candidates.sum(dim=-1) > self.min_fib_w, axis=1)
			fsb_2 = torch.all(candidates.view(nsamples, self.num_mix*self.num_comp) > self.min_comp_w, axis=1)
			fsb = fsb_1 & fsb_2
			if fsb.any():
				samples[fsb,...] = candidates[fsb,...]
				done = done | fsb
		return samples

	def log_prob(self, theta: Tensor) -> Tensor:
		return self.dir_dist.log_prob(theta)

class Discrepancy(Distribution):
	"""
	radial-angular sperable GP model for model misspecification term
	"""
	def __init__(self, q_space_design_orient, q_space_design_radial, angular_kernel, radial_kernel, sigma2, epsilon=1e-5):
		"""
		q_space_design_orient: nverts x 3 (dense design on the unit-sphere for computing covariance function)
		q_space_design_radial: nshells x 3 (number of b-value shells to sample random functions) 
		angular_kernel: orientational correlation
		radial_kernel: b-value correlation
		sigma2: nugget
		"""
		self.q_space_design_orient = q_space_design_orient
		self.n_desig_verts = self.q_space_design_orient.shape[0]
		self.q_space_design_radial = q_space_design_radial
		self.n_shells = len(self.q_space_design_radial)
		self.angular_kernel = angular_kernel
		self.radial_kernel = radial_kernel
		self.sigma2 = sigma2
		self.epsilon = epsilon ##numerical stability 
		batch_shape = torch.Size()
		super().__init__(batch_shape, validate_args=False)

	def _compute_covariance(self, X1, b1, X2, b2):
		Cov_X = self.angular_kernel(X1, X2)
		Cov_b = self.radial_kernel(b1, b2)
		return Cov_X * Cov_b ## radial-angular separability 

	def rsample(self, sample_shape=torch.Size(), seed: Optional[int] = None) -> Tensor:
		shape = self._extended_shape(sample_shape)
		nsamples = shape.numel()
		## build covariance matrix 
		## assemble block structure, torch does not have dedicated blocking function, map to numpy and use theirs
		blocks = [[None]*self.n_shells for x in range(self.n_shells)]
		for i in range(self.n_shells):
			bi = self.q_space_design_radial[i]*torch.ones(self.n_desig_verts,1)
			for j in range(i, self.n_shells):
				bj = self.q_space_design_radial[j]*torch.ones(self.n_desig_verts,1)
				Sigma_delta_ij = self._compute_covariance(self.q_space_design_orient, bi, self.q_space_design_orient, bj).cpu().detach().numpy()
				blocks[i][j] = Sigma_delta_ij
				if i != j:
					blocks[j][i] = Sigma_delta_ij
		Sigma_delta = self.sigma2 * torch.from_numpy(np.block(blocks)).float()
		Delta = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.n_shells*self.n_desig_verts), Sigma_delta + torch.diag(self.epsilon*torch.ones(Sigma_delta.shape[0])))
		Samples_grid_flat = Delta.sample((nsamples,)) ##nsamples x self.n_shells*self.n_desig_verts
		## map the function sample evaluations to each shell 
		Samples_grid_byshell = torch.zeros(nsamples, self.n_shells, self.n_desig_verts)
		for i in range(self.n_shells):
			Samples_grid_byshell[:, i, :] = Samples_grid_flat[:, (i*self.n_desig_verts):((i+1)*self.n_desig_verts)]
		return Samples_grid_byshell

	def log_prob(self, theta: Tensor) -> Tensor:
		raise NotImplementedError



