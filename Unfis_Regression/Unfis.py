import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Unfis(nn.Module):
    def __init__(self, membfuncs: list, n_input: int, to_device: Optional[str] = None):
        super(Unfis, self).__init__()
        self.name = 'Unfis-R'
        self.rules = membfuncs[0]['n_memb']
        self._n = n_input
        # build model
        self.layers = nn.ModuleDict({
            # Layer 1 - fuzzyfication
            'fuzzylayer': _FuzzyLayer(membfuncs, self._n, self.rules),
            # Layer 2 - selection
            'selection': _selectionLayer(self._n, self.rules),
            # Layer 3 - rule layer
            'rules': _RuleLayer(),
            # Layer 4 - normalization - is a simple function --> see forward pass
            # Layer 5 - consequence layer
            'consequence': _ConsequenceLayer(self._n, self.rules),
            # Layer 6 - weighted-sum - is a simple function
        })

        # determine device (cuda / cpu) if not specifically given
        if to_device == None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        else:
            self.device = to_device
        print("Device is : ", self.device)
        self.to(self.device)

    # Network architecture is defined in terms of class properties
    # You shall not switch the architecture after creating an object of SANFIS
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        # Layer 1 - fuzzyfication
        output1 = self.layers['fuzzylayer'](X_batch)
        output2, s = self.layers['selection'](output1)
        # Layer 2 - rule layer
        output3 = self.layers['rules'](output2)
        # Layer 3 - normalization layer // output3 == wnorm
        output4 = F.normalize(output3, p=1, dim=1)
        # Layer 4 - consequence layer
        output5 = self.layers['consequence'](X_batch, output4, s)
        # Layer 5 - summation
        output6 = output5.sum(axis=1).reshape(-1, 1)
        return output6

    def _reset_model_parameters(self):
        """reset model parameters (for early stopping procedure)
        """
        optlcass = self.optimizer.__class__
        self.optimizer = optlcass(self.parameters(), lr=self.optimizer.__dict__['param_groups'][0]['lr'])

        # reset parameters
        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


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
    def __init__(self, n_input, n_rules):
        super(_ConsequenceLayer, self).__init__()
        self.n = n_input
        self.rules = n_rules

        # weights
        self._weight = nn.Parameter(torch.Tensor(self.n, n_rules))
        self._bias = nn.Parameter(torch.Tensor(1, n_rules))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self._weight[:] = torch.rand(
                self.n, self.rules) - 0.5

            self._bias[:] = torch.rand(1, self.rules) - 0.5

    def forward(self, input_, wnorm, s):
        output = wnorm * (torch.matmul(input_, self._weight * s) + self._bias)
        return output
