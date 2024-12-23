import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List, Optional, Union
from helpers import _FastTensorDataLoader


class ANFIS(nn.Module):
    def __init__(self, membfuncs: list, classes: int, n_input: int, to_device: Optional[str] = None):
        super(ANFIS, self).__init__()
        self.name = 'Unfis-C'
        self.rules = membfuncs[0]['n_memb']
        self._n = n_input
        self.classes = classes
        # build model
        self.layers = nn.ModuleDict({
            # Layer 1 - fuzzyfication
            'fuzzylayer': _FuzzyLayer(membfuncs,self._n,self.rules),
            # Layer 2 - selection
            'selection': _selectionLayer(self._n, self.rules),
            # Layer 3 - rule layer
            'rules': _RuleLayer(),
            # Layer 4 - normalization - is a simple function --> see forward pass
            # Layer 5 - reconstructor - is a simple mlp to reconstruct input
            'mlp': MLP(self.rules,self._n),
            # Layer 5 - consequence layer
            'consequence': _ConsequenceLayer(self._n,self.rules, self.classes),
            # Layer 6 - a simple classifier
        })
        # determine device (cuda / cpu) if not specifically given
        if to_device == None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = to_device
        self.to(self.device)
        print("Device is : ", self.device)

    # Network architecture is defined in terms of class properties
    # You shall not switch the architecture after creating an object of SANFIS
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:

        # Layer 1 - fuzzyfication
        output1 = self.layers['fuzzylayer'](X_batch)
        # Layer 2 - selection
        output2, s = self.layers['selection'](output1)
        # Layer 3 - rule layer
        output3 = self.layers['rules'](output2)
        # Layer 4 - normalization layer // output3 == wnorm
        output4 = F.normalize(output3, p=1, dim=1)
        # Layer 5 - reconstructor
        reconstruct = self.layers['mlp'](output4)
        # Layer 6 - consequence layer
        output5 = self.layers['consequence'](X_batch, output4, s)
        # Layer 7 - normalize based on rules
        output6 = F.normalize(torch.sum(output5, 1), p=1, dim=1)
        return output6,reconstruct

    def _reset_model_parameters(self):
        """reset model parameters (for early stopping procedure)
        """
        print("resetttttt")
        optlcass = self.optimizer.__class__
        self.optimizer = optlcass(self.parameters(), lr=self.optimizer.__dict__[
            'param_groups'][0]['lr'])

        # reset parameters
        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

class MLP(nn.Module):

    def __init__(self,input_dim, num_outputs):

        super(MLP, self).__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(inplace=True),
            nn.Linear(12, num_outputs)
        )
        # Initialize FC layer weights using He initialization
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, x):
        output = self.FC(x)
        return output
class _FuzzyLayer(nn.Module):
    def __init__(self, membfuncs, n, r):
        """Represents the fuzzy layer (layer 1) of anfis. Inputs will be fuzzyfied
        """
        self.n = n
        self.r = r
        super(_FuzzyLayer, self).__init__()
        self._mu = torch.Tensor(n, r)
        self._sigma = torch.Tensor(n, r)
        for i in range(len(membfuncs)):
            self._mu[i] = torch.tensor(membfuncs[i]['params']['mu']['value'])
            self._sigma[i] = torch.tensor(membfuncs[i]['params']['sigma']['value'])
        self._mu = nn.Parameter(self._mu)
        self._sigma = nn.Parameter(self._sigma)

    def reset_parameters(self):
        self._mu = nn.Parameter(torch.Tensor(self.n, self.r))
        self._sigma = nn.Parameter(torch.Tensor(self.n, self.r))

    def forward(self, input_):
        output = torch.exp(
            - torch.square(
                (input_.unsqueeze(2).repeat(1, 1, self.r)) - self._mu)
            / self._sigma.square()
        )
        return output


class _RuleLayer(nn.Module):
    def __init__(self):
        super(_RuleLayer, self).__init__()

    def forward(self, input_):
        output = torch.min(input_, 1)[0]
        return output


class _selectionLayer(nn.Module):
    def __init__(self, n_input, n_rules):
        super(_selectionLayer, self).__init__()
        self._s = nn.Parameter(torch.full((n_input, n_rules), 1.5))
        self.eps = torch.full((1, n_rules), 0.000001).to("cuda:0")

    def forward(self, input_):
        theta = 1 / (1 + torch.exp(-self._s))
        output = (input_ + self.eps) / (((1 - theta) * input_) + theta + self.eps)
        return output, self._s


class _ConsequenceLayer(nn.Module):
    def __init__(self, n_input, n_rules, n_classes):
        super(_ConsequenceLayer, self).__init__()
        self.n = n_input
        self.rules = n_rules
        self.n_classes = n_classes
        # weights
        self.weights = nn.Parameter(torch.Tensor(self.n, self.rules, n_classes).to("cuda:0"))
        self.biases = nn.Parameter(torch.Tensor(1, self.rules, n_classes).to("cuda:0"))
        self.reset_parameters()
    def reset_parameters(self):
        with torch.no_grad():
            self.weights[:] = torch.rand(self.n, self.rules, self.n_classes) - 0.5
            self.biases[:] = torch.rand(1, self.rules, self.n_classes) - 0.5
    def forward(self, input_, wnorm, s):
        output = wnorm.unsqueeze(2) * (torch.einsum('ij,jkl->ikl', input_, s.unsqueeze(2) * self.weights) + self.biases)
        return output
