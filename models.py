import torch 
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, input_channels, input_size, hidden_sizes, output_size, activation=nn.ReLU(), dropout=0.0):
		super().__init__()
		layers = []
		# Add the first linear layer to handle input channels
		layers.append(nn.Linear(input_channels * input_size, hidden_sizes[0]))
		layers.append(activation)
		if dropout > 0.0:
			layers.append(nn.Dropout(dropout))
		for i in range(1, len(hidden_sizes)):
			layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
			layers.append(activation)
			if dropout > 0.0:
				layers.append(nn.Dropout(dropout))
		layers.append(nn.Linear(hidden_sizes[-1], output_size))
		self.mlp = nn.Sequential(*layers)
	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.mlp(x)
		
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