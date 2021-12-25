import torch
from configuration import config

class Policy(torch.nn.Module):


    def __init__(self, layers, nonlinearities, act_limits=1., use_bias=True,
                 name='neural_network'):

        super(Policy, self).__init__()

        self.layers = layers
        self.nonlinearities = nonlinearities
        self.use_bias = use_bias

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

        self.pi = self.mlp(layers, nonlinearities)

        self.act_limits = act_limits

    def forward(self, states):
       return self.pi(states)
  

    def mlp(self, sizes, activations):
        layers = []
        for j in range(len(sizes)-2):
            layers += [torch.nn.Linear(sizes[j], sizes[j+1], bias=self.use_bias, dtype=config.dtype), activations[j]()]
        
        layers += [torch.nn.Linear(sizes[-2], sizes[-1], bias=False, dtype=config.dtype), activations[-1]()]
        return torch.nn.Sequential(*layers)

    def lipschitz(self):
        lipschitz = torch.tensor(1, dtype=config.dtype)

        for i, param in enumerate(self.parameters()):
            if self.use_bias and i%2:
                pass
        else:
            lipschitz *= torch.max(torch.svd(param)[1])
 
        return lipschitz
