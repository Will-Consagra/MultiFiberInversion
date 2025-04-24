import torch 
import torch.nn as nn

class OperatorNetworkDirect(nn.Module):
	"""
	Orientation Inverter
	"""
	def __init__(self, summary_network, minval=-40, maxval=10):
		super().__init__()
		self._hidden_net = summary_network
	def forward(self,  context):
		"""
		context: Tensor of observed signal functions 
		"""
		hidden_net_results = self._hidden_net(context)
		h_equi = hidden_net_results["model_out_equi"] ##nbatch X n_equi X self._hidden_net.nvertices
		v = torch.sum(h_equi, axis=1) 
		return v