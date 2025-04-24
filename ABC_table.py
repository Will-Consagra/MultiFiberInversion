import numpy as np 
import scipy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

import time 

import pickle 

import sys 
import os 

from multi_fiber_inversion.statistical_models.embedding import SphericalSummaryStatistic
from multi_fiber_inversion.statistical_models.functional_operators import HarmonicRegression, sph_harm_ind_list, matern_spec_density, real_sym_sh_basis
from multi_fiber_inversion.distributions.distribution_objects import UniformAxial, BoxUniform, ConvexCombination
from multi_fiber_inversion.utility.helper_functions import S2hemisphere, cart2sphere
from multi_fiber_inversion.utility.sampling_S2 import HealpixSampling
from multi_fiber_inversion.local_models.stationary_kernel import SimulatorObjectSMDirect

from utility import smooth_pinv, sample_constrained_prior, NoisyMeasurementOperator

PATH2MFI = "multi_fiber_inversion"
TRAINDIR = "ABC_table"

## make reproducible
torch.manual_seed(0)

## get gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Define S2 geometry + function space  
## define function space 
sh_order = 8 
K = int((sh_order+1)*(sh_order+2)/2)
m, n = sph_harm_ind_list(sh_order) 
## graph structure to approximate S2 manifold 
n_side = 16
depth = 5
graphSampling = HealpixSampling(n_side, depth, sh_degree=sh_order)
pooling = graphSampling.pooling
laps = graphSampling.laps
K_input, V_input = graphSampling.sampling.SH2S.shape
design = graphSampling.sampling.vectors
design_tensor = torch.from_numpy(design).float().to(device) ## need to evaluate coefficinet vector on this grid 

#### Define priors 

## shape prior 
max_r = 0.8; min_r = 0
param_ranges = torch.from_numpy(np.array([(0.2,3), #D_a
										  (0.2,3), #D_e_parallel
										  (min_r,max_r), #r
										  (0.1, 0.9), ## z
						]).astype(np.float32).T)

shape_prior = BoxUniform(
						  low=param_ranges[0,:],
						  high=param_ranges[1,:]
						  )

## orientational prior 
orient_prior = UniformAxial()

## weight prior (fixed number of fibers)
max_fibers = 3
min_weight = 0.2

## high-conentration Watson mixture model for fodf
kappa_fodf = torch.tensor(40.) ## add a little smoothness to the dirac mixture 

## contrain the prior 
min_crossing_angle = 0.15 ## around 10 degrees 
min_perp_crossing = 0.52 ## if any two angles are within min_perp_crossing[0] radians, then any additional perpendicular fiber must be > min_perp_crossing[1] radians (prevents a bouquet pattern)

#### Build ABC Simulation table  

## acquisition pramaters
bvals = (1., 3) ##units ms/\mu m ^2
simulator_b0 = SimulatorObjectSMDirect((bvals[0],))
simulator_b1 = SimulatorObjectSMDirect((bvals[1],))
design_b0 = torch.load(os.path.join(PATH2MFI, "data", "q_space_design_hcp_ep_b1500.pt")) 
design_b1 = torch.load(os.path.join(PATH2MFI, "data", "q_space_design_hcp_ep_b3000.pt")) 
M = design_b0.shape[0]

## resampling grid 
HS2_grid = torch.load(os.path.join(PATH2MFI, "data", "H2_ESR_grid_300.pt"))
X_sph_grid = cart2sphere(HS2_grid)
theta_grid = X_sph_grid[:,0]; phi_grid = X_sph_grid[:,1]
Phi_grid = real_sym_sh_basis(sh_order, phi_grid, theta_grid)
Phi_grid_tensor = torch.from_numpy(Phi_grid).float().to(device)

## define distribution of random discrepancy
## measurement error process 
sigma2_e_b0 = sigma2_e_b1 = 0.0038421443951290267 ## this should be estimated based on your acquisition 

## assume white noise for measurement process
Sigma_inv_b0 = torch.eye(design_b0.shape[0]).to(device) 
Sigma_inv_b1 = torch.eye(design_b1.shape[0]).to(device)

## roughness penalty strength (this will depend on M and sigma2)
lambda_c_0 = 1e-3
lambda_c_1 = 1e-3

## noisy measurement operator 
measurement_b0 = NoisyMeasurementOperator(design_b0, Sigma_inv_b0, sigma2_e_b0, lambda_c=lambda_c_0)
measurement_b1 = NoisyMeasurementOperator(design_b1, Sigma_inv_b1, sigma2_e_b1, lambda_c=lambda_c_1)

## GAM fitting parameters resampling grid 
num_tilde = 100 
marg_rank = 20

## sample from forward model 
Ntrain = 5000000
random_fiber_mixtures = torch.randint(low=1, high=max_fibers+1, size=(Ntrain,))
num_fibers, num_fiber_counts = torch.unique(random_fiber_mixtures, return_counts=True)
num_fibers = list(map(lambda e: int(e), num_fibers.cpu().detach().numpy()))
num_fiber_counts = list(map(lambda e: int(e), num_fiber_counts.cpu().detach().numpy()))

Signals_coef_map = {}; Signals_obs_map = {}; Signals_resampled_map = {};
tilde_obs_map = {}; tilde_resampled_map = {};
direct_map = {}; shape_map={}; weight_map = {}; xi_map = {};
psi_map = {}
for nf, nf_count in zip(num_fibers, num_fiber_counts):
	Signal_coefs = torch.zeros(nf_count, len(bvals), K).to(device); Signal_obs = torch.zeros(nf_count, len(bvals), M).to(device)
	if nf > 1:
		weight_prior_nf = ConvexCombination(num_mix=nf, min_weight=min_weight)
		directs_nf, shapes_nf = sample_constrained_prior(nf_count, 
												 nf, 
												 min_crossing_angle=min_crossing_angle,
												 min_perp_crossing=min_perp_crossing,
												 max_r=max_r)
		## sample weights 
		weights_nf = weight_prior_nf.sample((nf_count,)) 
	else:
		weights_nf = torch.ones((nf_count,1))
		directs_nf = orient_prior.sample((nf_count,)).unsqueeze(1)
		shapes_nf = shape_prior.sample((nf_count,))
		shapes_nf[:,2] = shapes_nf[:,2]*shapes_nf[:,1] ##impose constraint: D_e_perp= r*D_e_parallel
		shapes_nf = shapes_nf.unsqueeze(1)
	## sample true signals
	simulator_out_nf_b0 = simulator_b0(design_b0, directs_nf, weights_nf, shapes_nf)
	simulator_out_nf_b1 = simulator_b1(design_b1, directs_nf, weights_nf, shapes_nf)
	## apply noisy measurement operator 
	mu_sig_b0, noisy_sig_b0 = measurement_b0(simulator_out_nf_b0["Stensor"][:,0,:].to(device))
	mu_sig_b1, noisy_sig_b1 = measurement_b1(simulator_out_nf_b1["Stensor"][:,0,:].to(device))
	Signal_coefs[:, 0, :] = mu_sig_b0
	Signal_coefs[:, 1, :] = mu_sig_b1
	Signal_obs[:, 0, :] = noisy_sig_b0
	Signal_obs[:, 1, :] = noisy_sig_b1
	Signal_resampled = Signal_coefs @ Phi_grid_tensor.T
	## get \tilde{t}_{i,m}^2
	tilde_nf_b0 = directs_nf @ design_b0.T 
	tilde_nf_b1 = directs_nf @ design_b1.T 
	tilde_nf = torch.stack((tilde_nf_b0, 
									tilde_nf_b1), dim=1)
	tilde_nf_grid = directs_nf @ HS2_grid.T 
	## optimize separately over z*w, (1-z)*w
	xi_nf = torch.cat((shapes_nf[...,:3], (shapes_nf[...,3]*weights_nf).unsqueeze(dim=-1), ((1-shapes_nf[...,3])*weights_nf).unsqueeze(dim=-1)), dim=-1)
	## save data 
	tilde_obs_map[nf] = torch.square(tilde_nf).cpu().detach().numpy()
	tilde_resampled_map[nf] = torch.square(tilde_nf_grid).cpu().detach().numpy()
	Signals_coef_map[nf] = Signal_coefs.cpu().detach().numpy()
	Signals_obs_map[nf] = Signal_obs.cpu().detach().numpy()
	Signals_resampled_map[nf] = Signal_resampled.cpu().detach().numpy()
	direct_map[nf] = directs_nf.cpu().detach().numpy()
	shape_map[nf] = shapes_nf.cpu().detach().numpy()
	weight_map[nf] = weights_nf.cpu().detach().numpy()
	xi_map[nf] = xi_nf.cpu().detach().numpy()
	psi_nf = np.zeros((nf_count, 2, nf, marg_rank-1))
	for i in range(nf_count):
		if nf == 1: ## approx 0.016 seconds per i 
			tilde_1_bv0_i = tilde_resampled_map[1][i, 0, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_1_bv1_i = tilde_resampled_map[1][i, 0, :]
			sig_bv0_i = Signals_resampled_map[1][i, 0, :]
			sig_bv1_i = Signals_resampled_map[1][i, 1, :]
			spline_basis_bv0_i = BSplines(tilde_1_bv0_i, df=marg_rank, degree=3, constraints="center")
			spline_basis_bv1_i = BSplines(tilde_1_bv1_i, df=marg_rank, degree=3, constraints="center")
			gam_bv0_i = GLMGam(sig_bv0_i, smoother=spline_basis_bv0_i)
			gam_bv1_i = GLMGam(sig_bv1_i, smoother=spline_basis_bv1_i)
			gam_results_bv0_i = gam_bv0_i.fit()
			gam_results_bv1_i = gam_bv1_i.fit()
			psi_nf[i, 0, 0, :] = gam_results_bv0_i.params
			psi_nf[i, 1, 0, :] = gam_results_bv1_i.params
		elif nf == 2: ## approx 0.048 seconds per i 
			tilde_1_bv0_i = tilde_resampled_map[2][i, 0, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_1_bv1_i = tilde_resampled_map[2][i, 0, :]
			tilde_2_bv0_i = tilde_resampled_map[2][i, 1, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_2_bv1_i = tilde_resampled_map[2][i, 1, :]
			sig_bv0_i = Signals_resampled_map[2][i, 0, :]
			sig_bv1_i = Signals_resampled_map[2][i, 1, :]
			spline_basis_bv0_i = BSplines(np.column_stack([tilde_1_bv0_i, tilde_2_bv0_i]), df=[marg_rank, marg_rank], degree=[3, 3], constraints="center")
			spline_basis_bv1_i = BSplines(np.column_stack([tilde_1_bv1_i, tilde_2_bv1_i]), df=[marg_rank, marg_rank], degree=[3, 3], constraints="center")
			gam_bv0_i = GLMGam(sig_bv0_i, smoother=spline_basis_bv0_i)
			gam_bv1_i = GLMGam(sig_bv1_i, smoother=spline_basis_bv1_i)
			gam_results_bv0_i = gam_bv0_i.fit()
			gam_results_bv1_i = gam_bv1_i.fit()
			psi_nf[i, 0, 0, :] = gam_results_bv0_i.params[:(marg_rank-1)]
			psi_nf[i, 0, 1, :] = gam_results_bv0_i.params[(marg_rank-1):]
			psi_nf[i, 1, 0, :] = gam_results_bv1_i.params[:(marg_rank-1)]
			psi_nf[i, 1, 1, :] = gam_results_bv1_i.params[(marg_rank-1):]
		elif nf == 3: ## approx 0.081 seconds per i 
			tilde_1_bv0_i = tilde_resampled_map[3][i, 0, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_1_bv1_i = tilde_resampled_map[3][i, 0, :]
			tilde_2_bv0_i = tilde_resampled_map[3][i, 1, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_2_bv1_i = tilde_resampled_map[3][i, 1, :]
			tilde_3_bv0_i = tilde_resampled_map[3][i, 2, :] ## bv0 = bv1 since we re-sample the signal onto the same high res grid
			tilde_3_bv1_i = tilde_resampled_map[3][i, 2, :]
			sig_bv0_i = Signals_resampled_map[3][i, 0, :]
			sig_bv1_i = Signals_resampled_map[3][i, 1, :]
			spline_basis_bv0_i = BSplines(np.column_stack([tilde_1_bv0_i, tilde_2_bv0_i, tilde_3_bv0_i]), df=[marg_rank, marg_rank, marg_rank], degree=[3, 3, 3], constraints="center")
			spline_basis_bv1_i = BSplines(np.column_stack([tilde_1_bv1_i, tilde_2_bv1_i, tilde_3_bv1_i]), df=[marg_rank, marg_rank, marg_rank], degree=[3, 3, 3], constraints="center")
			gam_bv0_i = GLMGam(sig_bv0_i, smoother=spline_basis_bv0_i)
			gam_bv1_i = GLMGam(sig_bv1_i, smoother=spline_basis_bv1_i)
			gam_results_bv0_i = gam_bv0_i.fit()
			gam_results_bv1_i = gam_bv1_i.fit()
			psi_nf[i, 0, 0, :] = gam_results_bv0_i.params[:(marg_rank-1)]
			psi_nf[i, 0, 1, :] = gam_results_bv0_i.params[(marg_rank-1):(2*(marg_rank-1))]
			psi_nf[i, 0, 2, :] = gam_results_bv0_i.params[(2*(marg_rank-1)):]
			psi_nf[i, 1, 0, :] = gam_results_bv1_i.params[:(marg_rank-1)]
			psi_nf[i, 1, 1, :] = gam_results_bv1_i.params[(marg_rank-1):(2*(marg_rank-1))]
			psi_nf[i, 1, 2, :] = gam_results_bv1_i.params[(2*(marg_rank-1)):]
	psi_map[nf] = psi_nf
	print("Finished Fiber ", nf)

print("... saving ABC table ...")

with open(os.path.join(TRAINDIR, "Signals_coef_map.pkl"), "wb") as pklfile:
	pickle.dump(Signals_coef_map, pklfile)

with open(os.path.join(TRAINDIR, "Signals_obs_map.pkl"), "wb") as pklfile:
	pickle.dump(Signals_obs_map, pklfile)

with open(os.path.join(TRAINDIR, "Signals_resampled_map.pkl"), "wb") as pklfile:
	pickle.dump(Signals_resampled_map, pklfile)

with open(os.path.join(TRAINDIR, "tilde_obs_map.pkl"), "wb") as pklfile:
	pickle.dump(tilde_obs_map, pklfile)

with open(os.path.join(TRAINDIR, "tilde_resampled_map.pkl"), "wb") as pklfile:
	pickle.dump(tilde_resampled_map, pklfile)

with open(os.path.join(TRAINDIR, "direct_map.pkl"), "wb") as pklfile:
	pickle.dump(direct_map, pklfile)

with open(os.path.join(TRAINDIR, "shape_map.pkl"), "wb") as pklfile:
	pickle.dump(shape_map, pklfile)

with open(os.path.join(TRAINDIR, "weight_map.pkl"), "wb") as pklfile:
	pickle.dump(weight_map, pklfile)

with open(os.path.join(TRAINDIR, "xi_map.pkl"), "wb") as pklfile:
	pickle.dump(xi_map, pklfile)

with open(os.path.join(TRAINDIR, "psi_map.pkl"), "wb") as pklfile:
	pickle.dump(psi_map, pklfile)



