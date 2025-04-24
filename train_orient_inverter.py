import numpy as np 
import scipy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dipy.direction import peak_directions
from dipy.data import get_sphere
from dipy.core.sphere import Sphere

import time 
from tqdm.autonotebook import tqdm

import pickle 

import sys 
import os 

from multi_fiber_inversion.statistical_models.embedding import SphericalSummaryStatistic
from multi_fiber_inversion.statistical_models.functional_operators import HarmonicRegression, sph_harm_ind_list, matern_spec_density, real_sym_sh_basis
from multi_fiber_inversion.distributions.distribution_objects import UniformAxial, BoxUniform, ConvexCombination
from multi_fiber_inversion.utility.helper_functions import S2hemisphere, cart2sphere
from multi_fiber_inversion.utility.sampling_S2 import HealpixSampling
from multi_fiber_inversion.local_models.stationary_kernel import SimulatorObjectSMDirect
from multi_fiber_inversion.local_models.fodfs import log_mixture_watson_fodf

from utility import smooth_pinv, sample_constrained_prior, NoisyMeasurementOperator, sample_constrained_prior_3fib_weights
from models import OperatorNetworkDirect

PATH2MFI = "multi_fiber_inversion"
TRAINDIR = "ABC_table"
MODELDIR = "models"

## make reproducible
torch.manual_seed(0)

## get gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Ntrain = 2000 ## number of simulations used in each round of training 

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
## create function space object 
FuncSpace = HarmonicRegression(device, design_tensor, sh_order=sh_order)
## orientational inference 
design_sphere = Sphere(xyz=design)
min_sep_angle = 10. ## in degreees
rel_pk_thrs = 0.55

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

min_degree_nf3 = 0.52
min_z1_weight = 0.1

#### Build ABC Simulation object 

## acquisition pramaters
bvals = (1.5, 3.) ##units ms/\mu m ^2
simulator_b0 = SimulatorObjectSMDirect((bvals[0],))
simulator_b1 = SimulatorObjectSMDirect((bvals[1],))
design_b0 = torch.load(os.path.join(PATH2MFI, "data", "q_space_design_hcp_ep_b1500.pt")) 
design_b1 = torch.load(os.path.join(PATH2MFI, "data", "q_space_design_hcp_ep_b3000.pt"))

## define distribution of random discrepancy
## measurement error process 
sigma2_e_b0 = sigma2_e_b1 = 0.0038421443951290267

## assume white noise for measurement process
Sigma_inv_b0 = torch.eye(design_b0.shape[0])
Sigma_inv_b1 = torch.eye(design_b1.shape[0])

## coefficient prior 
sigma2_c_b0 = sigma2_c_b1 = 2e-1
rho_b0 = rho_b1 = 0.5
nu_b0 = nu_b1 = 2.0
#R_tensor = torch.from_numpy(np.power(n*(n+1),2)).float().to(device)
R_tensor_b0 = torch.from_numpy(np.diag(matern_spec_density(np.sqrt(n*(n+1)), rho_b0, nu_b0))).float().to(device) ##precision matrix 
R_tensor_b1 = torch.from_numpy(np.diag(matern_spec_density(np.sqrt(n*(n+1)), rho_b1, nu_b1))).float().to(device) ##precision matrix 

lambda_c_0 = 1e-3
lambda_c_1 = 1e-3

measurement_b0 = NoisyMeasurementOperator(design_b0, Sigma_inv_b0, sigma2_e_b0, lambda_c=lambda_c_0)
measurement_b1 = NoisyMeasurementOperator(design_b1, Sigma_inv_b1, sigma2_e_b1, lambda_c=lambda_c_1)


#### Amortized Posterior 
## define summary statistic network fixed parameters 
### number of b-shells
in_channels = len(bvals)
### number of features that will be fed into the density estimator 
num_equi = 30
num_invar = 0 ## orientations should be invariant to all spherically invariant features ##  --> true, but uncertianties may not be?
## same parameterization as used in IPMI paper
filter_start = 8
kernel_size = 3

## optimization parameters  
learning_rate = 1e-4  
## build models 
## initialize summary network
summary_network = SphericalSummaryStatistic(in_channels, num_equi, 
											num_invar, V_input, 
											filter_start, kernel_size, 
											pooling, laps).to(device) ##add a channel for harmonic encoding of the 
## intialize posterior emulator
operator_emulator = OperatorNetworkDirect(summary_network).to(device)

## get function space for representation 
posterior_sh_order = 20
posterior_funcspace = HarmonicRegression(device, design_tensor, sh_order=posterior_sh_order)

## optimization algorithm  
optim = torch.optim.Adam(params=operator_emulator.parameters(), 
							lr=learning_rate)

num_epochs = 50000  

train_losses = []; test_evaluations = {}
suffix = "_bvals_%s_%s"%(Ntrain, bvals[0], bvals[1])
for epoch in range(num_epochs):
	results_epoch = {}
	start_time = time.time()
	random_fiber_mixtures = torch.randint(low=1, high=max_fibers+1, size=(Ntrain,))
	num_fibers, num_fiber_counts = torch.unique(random_fiber_mixtures, return_counts=True)
	num_fibers = list(map(lambda e: int(e), num_fibers.cpu().detach().numpy()))
	num_fiber_counts = list(map(lambda e: int(e), num_fiber_counts.cpu().detach().numpy()))
	Signals = torch.zeros(Ntrain, len(bvals), K).to(device); log_fodfs = torch.zeros(Ntrain, V_input).to(device)
	for nf, nf_count in zip(num_fibers, num_fiber_counts):
		## sample model parameters from prior 
		directs_nf, shapes_nf, weights_nf = sample_constrained_prior_3fib_weights(nf, nf_count, min_weight=min_weight, min_crossing_angle=min_crossing_angle, 
																						min_perp_crossing = min_perp_crossing, max_r=max_r, 
																						min_degree_nf3=min_degree_nf3, min_z1_weight=min_z1_weight)
		## sample true signals
		simulator_out_nf_b0 = simulator_b0(design_b0, directs_nf, weights_nf, shapes_nf)
		simulator_out_nf_b1 = simulator_b1(design_b1, directs_nf, weights_nf, shapes_nf)
		## apply noisy measurement operator 
		#mu_sig_b0, var_sig_b0 = measurement_b0(simulator_out_nf_b0["Stensor"][:,0,:].to(device))
		#mu_sig_b1, var_sig_b1 = measurement_b1(simulator_out_nf_b1["Stensor"][:,0,:].to(device))
		mu_sig_b0 = measurement_b0(simulator_out_nf_b0["Stensor"][:,0,:].to(device))
		mu_sig_b1 = measurement_b1(simulator_out_nf_b1["Stensor"][:,0,:].to(device))
		## compute fodfs under the highly concentrated Watson model
		kappas_fodfs_nf = kappa_fodf*torch.ones(nf_count,1)
		log_fodfs_nf = log_mixture_watson_fodf(design_tensor, directs_nf.to(device), kappas_fodfs_nf.to(device), num_t=100)
		## save mixture signals and density function
		ix_nf = torch.where(random_fiber_mixtures == nf)[0]
		Signals[ix_nf, 0, :] = mu_sig_b0
		Signals[ix_nf, 1, :] = mu_sig_b1
		log_fodfs[ix_nf, ...] = log_fodfs_nf.to(device)
		v = operator_emulator(FuncSpace(Signals))
		loss = ((v - torch.exp(log_fodfs))**2).mean() 
	optim.zero_grad()
	loss.backward()
	optim.step()
	train_losses.append(loss.item())
	## take off gpu
	Signals = Signals.cpu().detach()
	log_fodfs = log_fodfs.cpu().detach()
	#print("L2 loss:", loss.item())
	#print("Epoch %d, iteration time %0.3f, loss: %0.8f" % (epoch, time.time() - start_time, train_losses[-1]))
	if not (epoch%10000):
		torch.save(operator_emulator.state_dict(), os.path.join(MODELDIR, "orient_inverse_%s_%s.pth"%(suffix, epoch)))
		## create test set for performance tracking 
		Ntest = 2000
		random_fiber_mixtures_test = torch.randint(low=1, high=max_fibers+1, size=(Ntest,))
		num_fibers_test, num_fiber_counts_test = torch.unique(random_fiber_mixtures_test, return_counts=True)
		num_fibers_test = list(map(lambda e: int(e), num_fibers_test.cpu().detach().numpy()))
		num_fiber_counts_test = list(map(lambda e: int(e), num_fiber_counts_test.cpu().detach().numpy()))
		direct_map_test = {}; Signals_map_test = {}
		for nf, nf_count in zip(num_fibers_test, num_fiber_counts_test):
			Signals_test = torch.zeros(nf_count, len(bvals), K)
			## sample model parameters from prior 
			directs_nf, shapes_nf, weights_nf = sample_constrained_prior_3fib_weights(nf, nf_count, min_weight=min_weight, min_crossing_angle=min_crossing_angle, 
																							min_perp_crossing = min_perp_crossing, max_r=max_r, 
																							min_degree_nf3=min_degree_nf3, min_z1_weight=min_z1_weight)
			## sample true signals
			simulator_out_nf_b0 = simulator_b0(design_b0, directs_nf, weights_nf, shapes_nf)
			simulator_out_nf_b1 = simulator_b1(design_b1, directs_nf, weights_nf, shapes_nf)
			## apply noisy measurement operator 
			#mu_sig_b0, var_sig_b0 = measurement_b0(simulator_out_nf_b0["Stensor"][:,0,:].to(device))
			#mu_sig_b1, var_sig_b1 = measurement_b1(simulator_out_nf_b1["Stensor"][:,0,:].to(device))
			mu_sig_b0 = measurement_b0(simulator_out_nf_b0["Stensor"][:,0,:].to(device))
			mu_sig_b1 = measurement_b1(simulator_out_nf_b1["Stensor"][:,0,:].to(device))
			## compute fodfs under the highly concentrated Watson model
			kappas_fodfs_nf = kappa_fodf*torch.ones(nf_count,1)
			log_fodfs_nf = log_mixture_watson_fodf(design_tensor, directs_nf.to(device), kappas_fodfs_nf.to(device), num_t=100)
			## save mixture signals and density function
			Signals_test[:, 0, :] = mu_sig_b0
			Signals_test[:, 1, :] = mu_sig_b1
			Signals_map_test[nf] = Signals_test.cpu()
			direct_map_test[nf] = directs_nf
		#### run some evaluations ####
		LFI_ang_errors = {}; LFI_CP = {}
		LFI_ang_errors["t1"] = {}; LFI_CP["t1"] = {}
		LFI_ang_errors["t2"] = {}; LFI_CP["t2"] = {}
		for nf, nf_count in zip(num_fibers_test, num_fiber_counts_test):
			Signals_test_nf = Signals_map_test[nf].to(device)
			directs_test_nf = direct_map_test[nf]
			### infer fodf 
			with torch.no_grad():
				fodfs_hat_disc_batch = operator_emulator(FuncSpace(Signals_test_nf))
			### enforce anti-podal symmetry by projecting to symetric harmonic space 
			fodfs_shm_hat_batch = posterior_funcspace.S2C(fodfs_hat_disc_batch)
			#fodfs_shm_hat_batch = fodfs_shm_hat_batch / torch.norm(fodfs_shm_hat_batch, p=2, dim=-1)[...,None]
			fodfs_hat_batch = posterior_funcspace(fodfs_shm_hat_batch)
			fodfs_hat_batch = fodfs_hat_batch.cpu().detach()
			CP_LFI_nf = []; ang_errors_LFI_nf = {}
			CP_LFI_nf_v2 = []; ang_errors_LFI_nf_v2 = {}
			for nb in range(fodfs_hat_batch.shape[0]):
				pk_true_nb = directs_test_nf[nb,...]
				## Step 2: Model selection, orienation inference 
				dirs_nb, values_nb, indices_nb = peak_directions(fodfs_hat_batch[nb,:].numpy().astype("double"), 
																design_sphere, relative_peak_threshold=rel_pk_thrs, min_separation_angle=min_sep_angle)
				dirs_nb = S2hemisphere(dirs_nb)
				## calculate orientational accuracy 
				CP_LFI_nf.append(nf == dirs_nb.shape[0])
				if CP_LFI_nf[-1]:
					ang_errors_LFI_nf[nb] = compute_angular_error(pk_true_nb.unsqueeze(0), 
											torch.from_numpy(dirs_nb).float().unsqueeze(0)) 
				##### direction selection with much smaller threshold #####
				dirs_nb, values_nb, indices_nb = peak_directions(fodfs_hat_batch[nb,:].numpy().astype("double"), 
																design_sphere, relative_peak_threshold=0.4, min_separation_angle=min_sep_angle)
				dirs_nb = S2hemisphere(dirs_nb)
				## calculate orientational accuracy 
				CP_LFI_nf_v2.append(nf == dirs_nb.shape[0])
				if CP_LFI_nf_v2[-1]:
					ang_errors_LFI_nf_v2[nb] = compute_angular_error(pk_true_nb.unsqueeze(0), 
											torch.from_numpy(dirs_nb).float().unsqueeze(0)) 
			LFI_ang_errors["t1"][nf] = list(ang_errors_LFI_nf.values())
			LFI_CP["t1"][nf] = CP_LFI_nf
			LFI_ang_errors["t2"][nf] = list(ang_errors_LFI_nf_v2.values())
			LFI_CP["t2"][nf] = CP_LFI_nf_v2
			## take off gpu
			Signals_test_nf = Signals_test_nf.cpu().detach()
			del Signals_test_nf
			del fodfs_hat_batch
		#lfi_ang_errors_1 = np.mean([e.item() for e in LFI_ang_errors[1]])
		#lfi_ang_errors_2 = np.mean([e.item() for e in LFI_ang_errors[2]])
		#lfi_ang_errors_3 = np.mean([e.item() for e in LFI_ang_errors[3]])
		#lfi_ms_1 = np.mean(LFI_CP[1])
		#lfi_ms_2 = np.mean(LFI_CP[2])
		#lfi_ms_3 = np.mean(LFI_CP[3])
		## take off gpu
		print("Epoch %d, loss: %0.8f" % (epoch, train_losses[-1]))
		print("Threshold 1")
		print("Angular Errors:: 1: %0.3f, 2: %0.3f, 3: %0.3f", np.mean([e.item() for e in LFI_ang_errors["t1"][1]]), 
																np.mean([e.item() for e in LFI_ang_errors["t1"][2]]), 
																np.mean([e.item() for e in LFI_ang_errors["t1"][3]]))
		print("% Correct Model Selection:: 1: %0.3f, 2: %0.3f, 3: %0.3f",  np.mean(LFI_CP["t1"][1]), np.mean(LFI_CP["t1"][2]), np.mean(LFI_CP["t1"][3]))
		print("Threshold 2")
		print("Angular Errors:: 1: %0.3f, 2: %0.3f, 3: %0.3f", np.mean([e.item() for e in LFI_ang_errors["t2"][1]]), 
																np.mean([e.item() for e in LFI_ang_errors["t2"][2]]), 
																np.mean([e.item() for e in LFI_ang_errors["t2"][3]]))
		print("% Correct Model Selection:: 1: %0.3f, 2: %0.3f, 3: %0.3f",  np.mean(LFI_CP["t2"][1]), np.mean(LFI_CP["t2"][2]), np.mean(LFI_CP["t2"][3]))
		torch.cuda.empty_cache()  

	
