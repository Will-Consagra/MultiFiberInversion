import numpy as np 
import scipy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time 
from tqdm.autonotebook import tqdm

import pickle 

import sys 
import os 

from multi_fiber_inversion.statistical_models.functional_operators import HarmonicRegression, sph_harm_ind_list, matern_spec_density, real_sym_sh_basis
from multi_fiber_inversion.distributions.distribution_objects import UniformAxial, BoxUniform, ConvexCombination
from multi_fiber_inversion.utility.sampling_S2 import HealpixSampling
from multi_fiber_inversion.utility.helper_functions import S2hemisphere, cart2sphere
from multi_fiber_inversion.local_models.stationary_kernel import SimulatorObjectSMDirectReparam
from multi_fiber_inversion.pyknos.mdn import MultivariateGaussianMDN as GMM 

from models import MLP

PATH2MFI = "multi_fiber_inversion"
TRAINDIR = "ABC_table"
MODELDIR = "models"

## make reproducible
torch.manual_seed(0)

## get gpu if available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load saved ABC table 
bvals = (1.5, 3.)

#### 3-fiber-case ####
with open(os.path.join(TRAINDIR, "psi_map.pkl"), "rb") as pklfile:
	psi_map = pickle.load(pklfile)

Psi_hat_31_bv1 = psi_map[3][:, 0, 0, :]
Psi_hat_31_bv3 = psi_map[3][:, 1, 0, :]

Psi_hat_32_bv1 = psi_map[3][:, 0, 1, :]
Psi_hat_32_bv3 = psi_map[3][:, 1, 1, :]

Psi_hat_33_bv1 = psi_map[3][:, 0, 2, :]
Psi_hat_33_bv3 = psi_map[3][:, 1, 2, :]

Signals_31_hat = torch.cat((torch.from_numpy(Psi_hat_31_bv1).float().unsqueeze(1), 
						torch.from_numpy(Psi_hat_31_bv3).float().unsqueeze(1)), dim=1)
Signals_32_hat = torch.cat((torch.from_numpy(Psi_hat_32_bv1).float().unsqueeze(1), 
						torch.from_numpy(Psi_hat_32_bv3).float().unsqueeze(1)), dim=1)
Signals_33_hat = torch.cat((torch.from_numpy(Psi_hat_33_bv1).float().unsqueeze(1), 
						torch.from_numpy(Psi_hat_33_bv3).float().unsqueeze(1)), dim=1)
Signals_3 = torch.cat((Signals_31_hat, Signals_32_hat, Signals_33_hat), dim=0)

with open(os.path.join(TRAINDIR, "xi_map.pkl"), "rb") as pklfile:
	xi_map = pickle.load(pklfile)

theta_3_stacked = torch.from_numpy(xi_map[3]).float()
theta_3 = torch.cat((theta_3_stacked[:,0,:], theta_3_stacked[:,1,:], theta_3_stacked[:,2,:]), dim=0)

## inverse model 
input_channels = len(bvals)
p = 5
input_size = Signals_3.shape[-1]
num_summary_stats = 1000
num_hidden_layers = 10
hidden_sizes = [num_summary_stats]*num_hidden_layers
activation = nn.ReLU()
dropout = 0.0
summary_network = MLP(input_channels, input_size, hidden_sizes, num_summary_stats, activation=activation, dropout=dropout)
num_mixture_components = 5
inverse_model = GMM(p, input_size, num_summary_stats, summary_network, num_components=num_mixture_components, custom_initialization=False)
inverse_model = inverse_model.to(device)

## optimization algorithm  
learning_rate = 1e-5
optim = torch.optim.Adam(params=inverse_model.parameters(), 
								lr=learning_rate)
num_epochs = 5000; batch_size=10000;

## ABC simulation table 
dataset_abc = torch.utils.data.TensorDataset(theta_3, Signals_3)
N_abc = len(dataset_abc)
test_size = 5000
train_size = N_abc - test_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset_abc, [train_size, test_size])

dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
dataloader_test = DataLoader(test_dataset, batch_size=test_size, shuffle=False)  
for batch_test in dataloader_test:
	theta_test, Signals_test = batch_test

Signals_test = Signals_test.to(device)
theta_test = theta_test.to(device)

Npost_samps = 2000
test_losses = []; iter_ = 0;
suffix = "bvals_%s_%s"%(bvals[0], bvals[1])
#with tqdm(total=len(dataloader_train) * num_epochs) as pbar:
for epoch in range(num_epochs):
	start_time = time.time()
	for batch in dataloader_train:
		batch_theta, batch_Signals = batch
		batch_theta = batch_theta.to(device)
		batch_Signals = batch_Signals.to(device)
		log_lik = inverse_model.log_prob(batch_theta, batch_Signals)
		loss = -log_lik.mean()
		optim.zero_grad()
		loss.backward()
		optim.step()
		## compute testing errror 
		log_lik_test = inverse_model.log_prob(theta_test, Signals_test)
		loss_test = -log_lik_test.mean()
		test_losses.append(loss_test.mean().item())
		iter_ += 1
		#pbar.update(1)
		if not iter_ % 1000:
			torch.save(inverse_model.state_dict(), os.path.join(MODELDIR, "shape_inverse_3fib_%s_%s.pth"%(suffix, iter_)))
			## compute testing errror 
			log_lik_test = inverse_model.log_prob(theta_test, Signals_test)
			loss_test = -log_lik_test.mean()
			test_losses.append(loss_test.mean().item())
			print("Iteration %d, neg-log-lik: %0.8f" % (epoch, test_losses[-1]))
			## parameter inference 
			param_errors = []
			for n in range(Signals_test.shape[0]):
				signals_test_n = Signals_test[n:(n+1),...]
				theta_samples_n = inverse_model.sample(Npost_samps, signals_test_n).squeeze()  
				theta_hat_n = theta_samples_n.mean(dim=0).cpu().detach()
				param_errors.append((theta_test[n,:].cpu() - theta_hat_n).cpu().detach().numpy())
			param_errors = np.array(param_errors)
			print("Median L1 parameter errors")
			print(np.median(np.abs(param_errors), axis=0))


