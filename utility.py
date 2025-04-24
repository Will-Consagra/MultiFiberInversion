import numpy as np 
import scipy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multi_fiber_inversion.utility.helper_functions import cart2sphere
from multi_fiber_inversion.statistical_models.functional_operators import real_sym_sh_basis, sph_harm_ind_list


## utility functions 
def smooth_pinv(B, L): 
	"""Regularized pseudo-inverse

	Computes a regularized least square inverse of B

	Parameters
	----------
	B : array_like (n, m)
		Matrix to be inverted
	L : array_like (n,)

	Returns
	-------
	inv : ndarray (m, n)
		regularized least square inverse of B

	Notes
	-----
	In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
	However here this inverse is implemented using the pseudo-inverse because
	it is more numerically stable than the direct implementation of the matrix
	product.

	"""
	L = np.diag(L)
	inv = np.linalg.pinv(np.concatenate((B, L)))
	return inv[:, :len(B)]

def sample_constrained_prior(nsamples, num_fibers, min_crossing_angle=0.17, min_perp_crossing = 0.52, max_r=0.8):
	## Rejection sampling to obtain constrianed samples from the prior 
	done = torch.zeros((nsamples,), dtype=torch.bool)
	directs = torch.zeros(nsamples, num_fibers, 3)
	shapes = torch.zeros(nsamples, num_fibers, 4)
	tri_ix = torch.triu_indices(num_fibers, num_fibers, offset=1)
	while not done.all():
		directs_candidates = torch.zeros(nsamples, num_fibers, 3)
		shape_candidates = torch.zeros(nsamples, num_fibers, 4)
		for nf in range(num_fibers):
			directs_nf = orient_prior.sample((nsamples, ))
			shapes_nf = shape_prior.sample((nsamples,))
			shapes_nf[:,2] = shapes_nf[:,2]*shapes_nf[:,1] ##impose constraint: D_e_perp= r*D_e_parallel
			directs_candidates[:, nf, :] = directs_nf
			shape_candidates[:, nf, :] = shapes_nf
		## apply min crossing anlge constraints
		candidate_crossing_angles = torch.acos(torch.sqrt(torch.pow( torch.matmul(directs_candidates, directs_candidates.transpose(1,2)), 2) )) 
		candidate_crossing_angles_upper_triangle = torch.zeros(nsamples, num_fibers*(num_fibers-1)//2)
		for ns in range(nsamples):
			candidate_crossing_angles_upper_triangle[ns,:] = candidate_crossing_angles[ns, tri_ix[0], tri_ix[1]]
		if num_fibers == 2:
			crossing_angle_indicator = torch.where(torch.sum(candidate_crossing_angles_upper_triangle < min_crossing_angle, dim=1) > 0, False, True)
		else:
			candidate_crossing_angles_upper_triangle_sorted, _ = torch.sort(candidate_crossing_angles_upper_triangle, dim=1)
			CAI_1 = torch.where(candidate_crossing_angles_upper_triangle_sorted[:,0] < min_crossing_angle, False, True) ## crossing angle cannot be < min_crossing_angle
			CAI_2 = torch.where(candidate_crossing_angles_upper_triangle_sorted[:,1] < min_perp_crossing, False, True) ## smallest perpendicular angle cannot be < min_perp_crossing
			crossing_angle_indicator = CAI_1 & CAI_2
		##testing max isotropy shape constraint 
		isotropy_candidates = shape_candidates[:,:,2]/shape_candidates[:,:,1] ##D_e_perp/D_e_parallel = r
		isotropy_indicator = torch.where(torch.sum(isotropy_candidates > max_r, dim=1) > 0, False, True)
		##joint indicators (note, this can be sped up by separating the orientation and shape components due to independence, but for now this is fine since simulation is fast)
		accept_indicator = crossing_angle_indicator & isotropy_indicator
		if accept_indicator.any():
			directs[accept_indicator,:,:] = directs_candidates[accept_indicator,:,:]
			shapes[accept_indicator,:,:] = shape_candidates[accept_indicator,:,:]
			done = done | accept_indicator
	return directs, shapes

class NoisyMeasurementOperator(nn.Module):
	def __init__(self, design, Sigma_inv, sigma2_e, lambda_c=1e-3, sh_order=8, device="cpu"):
		super().__init__()
		self.design = design
		self.M = self.design.shape[0]
		self.Sigma_inv = Sigma_inv
		self.sigma2_e = sigma2_e
		X_sph = cart2sphere(design)
		theta_x = X_sph[:,0]; phi_x = X_sph[:,1]
		m, n = sph_harm_ind_list(sh_order)
		Phi = real_sym_sh_basis(sh_order, phi_x, theta_x)
		L = -n * (n + 1)
		device = torch.device(device)
		self.invPhi = torch.from_numpy(smooth_pinv(Phi, np.sqrt(lambda_c) * L)).float().to(device) 
		self.noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.zeros(self.M).to(device), precision_matrix=(1./sigma2_e)*self.Sigma_inv)
	def forward(self,  Signal_evals):
		"""
		context: Tensor of observed signal functions 
		"""
		White_noise = self.noise_dist.sample((Signal_evals.shape[0],))
		Noisy_Signal_evals = Signal_evals + White_noise
		mu_f = Noisy_Signal_evals @ self.invPhi.T  
		return mu_f, Noisy_Signal_evals
