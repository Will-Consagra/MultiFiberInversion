import numpy as np
import torch 
from scipy.special import legendre

### define forward model via Funk-Hecke derived integral equations 

## kernel for a single mixture component of the standard model
def StandardModel(t, b, D_a, D_e_parallel, D_e_perp, z_frac):
    ## note: this assumes D_e_perp= r*D_e_parallel for 0 < r < 1 
    return z_frac*torch.exp(-b*D_a*torch.pow(t, 2)) + (1-z_frac)*torch.exp(-b*D_e_perp - b*(D_e_parallel - D_e_perp)*torch.pow(t, 2))

def rho_k_SM(b, l_k, D_a, D_e_parallel, D_e_perp, z_frac, num_t=1000):
    """
    Use trapezoid rule, which has error O(num_t^{-2}) so we need to use a dense mesh 
    """
    t = torch.linspace(0, 1, steps=num_t).repeat(D_a.shape[0],1)
    kernel_func_evals = StandardModel(torch.linspace(0, 1, steps=num_t).repeat(D_a.shape[0],1),
                                            b, 
                                            D_a, 
                                            D_e_parallel, 
                                            D_e_perp, 
                                            z_frac)
    #legendre_evals = torch.special.legendre_polynomial_p(torch.linspace(0, 1, steps=num_t).repeat(D_a.shape[0],1), l_k)
    legendre_evals = legendre(l_k)(torch.linspace(0, 1, steps=num_t).repeat(D_a.shape[0],1))
    return torch.trapz(kernel_func_evals * legendre_evals, dx = 1/num_t).float()

class SimulatorObjectSM(torch.nn.Module):
    def __init__(self, bvals):
        super().__init__()
        self.bvals = bvals

    def forward(self, directs, weights, shapes, FuncSpace, num_t=1000):
        ## get globals 
        K = FuncSpace.K
        l_ks, l_k_indices = np.unique(FuncSpace.n, return_inverse=True)
        Nsamples = weights.shape[0]
        num_fibers = weights.shape[-1]

        ## storage results
        eigen_values = torch.zeros(Nsamples, len(self.bvals), num_fibers, K)
        ODFtensor = torch.zeros(Nsamples, num_fibers, K)
        Stensor_byfib = torch.zeros(Nsamples, len(self.bvals), num_fibers, K)

        ### run forward model
        for nf in range(num_fibers):
            ## get mixture components parameters 
            directs_nf =  directs[:, nf, :] 
            shapes_nf = shapes[:, nf, :]
            ## get eigenvalues 
            for ixk, l_k in enumerate(l_ks): 
                for ib, b in enumerate(self.bvals):
                    eignen_vals_nf = rho_k_SM(torch.tensor(b), torch.tensor(l_k), shapes_nf[:,0:1], shapes_nf[:,1:2], shapes_nf[:,2:3], shapes_nf[:,3:4], num_t=num_t)
                    eigen_values[:, ib, nf, np.where(l_k_indices == ixk)[0]] = eignen_vals_nf.repeat(np.sum(l_k_indices == ixk), 1).T
            ## get harmonic evaluations at orientations 
            ODFtensor[:, nf, :] = FuncSpace.harmonic_expansion(directs_nf)
            ## get signal function 
            for ib, b in enumerate(self.bvals):
                Stensor_byfib[:, ib, nf, :] = weights[:,nf:(nf+1)]*eigen_values[:, ib, nf, :]*ODFtensor[:, nf, :]
        ## sum mixture components 
        Stensor = torch.sum(Stensor_byfib, 2)
        return {"Stensor":Stensor, "ODFtensor":ODFtensor, "eigen_values":eigen_values}

class SimulatorObjectSMDirect(torch.nn.Module):
    def __init__(self, bvals):
        super().__init__()
        self.bvals = bvals

    def forward(self, design, directs, weights, shapes):
        ## get globals 
        Nsamples = weights.shape[0]
        num_fibers = weights.shape[-1]
        nverts = design.shape[0]

        ## storage results
        Stensor_byfib = torch.zeros(Nsamples, len(self.bvals), num_fibers, nverts)

        ### run forward model
        for nf in range(num_fibers):
            ## get mixture components parameters 
            directs_nf =  directs[:, nf, :] 
            shapes_nf = shapes[:, nf, :]
            ## inner product over design points 
            inner_products_nf = directs_nf @ design.T  ##Nsamples X nverts
            ## evaluate forward model
            for ib, b in enumerate(self.bvals):
            	fwd_nf_b = StandardModel(inner_products_nf, b, shapes_nf[:,0:1], shapes_nf[:,1:2], shapes_nf[:,2:3], shapes_nf[:,3:4])
            	Stensor_byfib[:, ib, nf, :] = weights[:,nf:(nf+1)]*fwd_nf_b
        ## sum mixture components 
        Stensor = torch.sum(Stensor_byfib, 2)
        return {"Stensor":Stensor}

class SimulatorObjectSMDirectReparam(torch.nn.Module):
    def __init__(self, bvals):
        super().__init__()
        self.bvals = bvals

    def _standard_model(self, t, b, D_a, D_e_parallel, D_e_perp, z_1, z_2):
        ## note: this assumes D_e_perp= r*D_e_parallel for 0 < r < 1 
        return z_1*torch.exp(-b*D_a*torch.pow(t, 2)) + z_2*torch.exp(-b*D_e_perp - b*(D_e_parallel - D_e_perp)*torch.pow(t, 2))

    def forward(self, design, directs, fracs, shapes):
        ## get globals 
        Nsamples = directs.shape[0]
        num_fibers = directs.shape[-2]
        nverts = design.shape[0]

        ## storage results
        Stensor_byfib = torch.zeros(Nsamples, len(self.bvals), num_fibers, nverts)

        ### run forward model
        for nf in range(num_fibers):
            ## get mixture components parameters 
            directs_nf =  directs[:, nf, :] 
            shapes_nf = shapes[:, nf, :]
            fracs_nf = fracs[:, nf, :]
            ## inner product over design points 
            inner_products_nf = directs_nf @ design.T  ##Nsamples X nverts
            ## evaluate forward model
            for ib, b in enumerate(self.bvals):
                fwd_nf_b = self._standard_model(inner_products_nf, b, shapes_nf[:,0:1], shapes_nf[:,1:2], shapes_nf[:,2:3], fracs_nf[:,0:1], fracs_nf[:,1:2])
                Stensor_byfib[:, ib, nf, :] = fwd_nf_b
        ## sum mixture components 
        Stensor = torch.sum(Stensor_byfib, 2)
        return {"Stensor":Stensor}


