import numpy as np
import torch 
from scipy.special import legendre

def mixture_dirac_fodf(function_space, directs):
	"""
	function_space: HarmonicRegression object defining the function space 
	spherical_grid: (Ngrid x 3)
	directs: (num_batch, num_fibers, 3)
	"""
	num_batch, num_components, _ = directs.shape
	Phi_evals = torch.zeros(num_batch, function_space.K, num_components)
	for nc in range(num_components):
		Phi_evals_nc = function_space.harmonic_expansion(directs[:, nc, :])
		Phi_evals[:, :, nc] = Phi_evals_nc
	return Phi_evals.mean(dim=-1)

def log_mixture_watson_fodf(Spherical_grid, directs, kappas, num_t=100):
	# Spherical_grid: (n_vertices, 3)
	# directs: (batch_size, num_fibers, 3)
	# kappa: (batch_size, 1), -> same for all fibers 

	batch_size, num_fibers, _ = directs.shape
	nverts, _ = Spherical_grid.shape

	##get normalization parameter (make sure this is a density function)
	t = torch.linspace(0, 1, steps=num_t).unsqueeze(0).expand((kappas.shape[0],num_t)).to(torch.device(kappas.device))
	C_kappa = 2*np.pi*torch.trapz(torch.exp(kappas * torch.pow(t,2)), dx = 1/num_t).float().to(torch.device(kappas.device))

	log_fodfs = torch.zeros(batch_size, num_fibers, nverts)

	for nf in range(num_fibers):

		directs_expanded_nf = directs[:,nf,:]
		inner_products_sqrd_nf = torch.pow((directs_expanded_nf @ Spherical_grid.T),2)

		term1 = torch.log(C_kappa).unsqueeze(1) 
		term2 = kappas * inner_products_sqrd_nf
		log_fodfs[:,nf,:] = term2 - term1

	return torch.logsumexp(log_fodfs, dim=1)

def mixture_watson_fodf(Spherical_grid, directs, kappas, num_t=100):
	# Spherical_grid: (n_vertices, 3)
	# directs: (batch_size, num_fibers, 3)
	# kappa: (batch_size, 1), -> same for all fibers 
	batch_size, num_fibers, _ = directs.shape
	nverts, _ = Spherical_grid.shape
	##get normalization parameter (make sure this is a density function)
	t = torch.linspace(0, 1, steps=num_t).unsqueeze(0).expand((kappas.shape[0],num_t)).to(torch.device(kappas.device))
	C_kappa = 2*np.pi*torch.trapz(torch.exp(kappas * torch.pow(t,2)), dx = 1/num_t).float().to(torch.device(kappas.device)).unsqueeze(1) 
	fodfs = torch.zeros(batch_size, num_fibers, nverts)
	for nf in range(num_fibers):
		directs_expanded_nf = directs[:,nf,:]
		inner_products_sqrd_nf = torch.pow((directs_expanded_nf @ Spherical_grid.T),2)
		watson_kernel = torch.exp(kappas * inner_products_sqrd_nf)
		fodfs[:,nf,:] = (1./num_fibers)*(1./C_kappa)*watson_kernel
	return torch.sum(fodfs, dim=1)