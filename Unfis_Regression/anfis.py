
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import List, Optional, Union
from helpers import  _FastTensorDataLoader, DataScaler


class ANFIS(nn.Module):
    def __init__(self, membfuncs: list,classes:int, n_input: int, to_device: Optional[str] = None, scale: str = 'None'):
        super(ANFIS, self).__init__()
        self.name = 'Unfis'
        self.pred = "train"
        self._membfuncs = membfuncs
        self._memberships = [memb['n_memb'] for memb in membfuncs]
        self._rules = int(np.prod(self._memberships))
        self._s = len(membfuncs)
        self._n = n_input
        self.scaler = DataScaler(scale)
        self.classes = classes
        # build model
        self.layers = nn.ModuleDict({
            # Layer 1 - fuzzyfication
            'fuzzylayer': _FuzzyLayer(membfuncs),

            # Layer 2 - rule layer
            'rules': _RuleLayer(),
            'selection': _selectionLayer(self._n,membfuncs[0]['n_memb']),
            # Layer 3 - normalization - is a simple function --> see forward pass

            # Layer 4 - consequence layer
            'consequence': _ConsequenceLayer(self._n, membfuncs[0]['n_memb']),
            # Layer 5 - weighted-sum - is a simple function
        })

        # save initial fuzzy weights
        self._initial_premise = self.premise

        # determine device (cuda / cpu) if not specifically given
        if to_device == None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = to_device

        self.to(self.device)

    # Network architecture is defined in terms of class properties
    # You shall not switch the architecture after creating an object of SANFIS
    def forward(self, S_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:

        # Layer 1 - fuzzyfication
        output1 = self.layers['fuzzylayer'](S_batch)
        output2,s = self.layers['selection'](output1)
        # Layer 2 - rule layer
        output2 = self.layers['rules'](output2)

        # Layer 3 - normalization layer // output3 == wnorm
        output3 = F.normalize(output2, p=1, dim=1)

        # Layer 4 - consequence layer
        output4 = self.layers['consequence'](
            X_batch, output3,s)

        # Layer 5 - summation
        output5 = output4.sum(axis=1).reshape(-1, 1)

        return output5

    def _reset_model_parameters(self):
        """reset model parameters (for early stopping procedure)
        """
        optlcass = self.optimizer.__class__
        self.optimizer = optlcass(self.parameters(), lr=self.optimizer.__dict__[
            'param_groups'][0]['lr'])

        # reset parameters
        with torch.no_grad():
            for layer in self.layers.values():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


    @property
    def n_statevars(self):
        return self._s

    @property
    def n_input(self):
        return self._n

    @property
    def memberships(self):
        return self._memberships

    @property
    def num_rules(self):
        return self._rules

    @property
    def premise(self):
        return [level.coeffs for level in self.layers.fuzzylayer.fuzzyfication]

    @premise.setter  # TODO: REFRESH
    def premise(self, new_memberships: list):
        self.layers.fuzzylayer = _FuzzyLayer(new_memberships)
        self._initial_premise = self.premise

    @property
    def consequence(self):
        return self.layers['consequence'].coeffs

    @consequence.setter
    def consequence(self, new_consequence: dict):
        self.layers['consequence'].coeffs = new_consequence

    @property
    def scaling_params(self):
        return self.scaler.scaler.__dict__


class _FuzzyLayer(nn.Module):
    def __init__(self, membfuncs):
        """Represents the fuzzy layer (layer 1) of anfis. Inputs will be fuzzyfied
        """
        super(_FuzzyLayer, self).__init__()
        self.n_statevars = len(membfuncs)

        fuzzyfication = nn.ModuleList()

        for membfunc in membfuncs:
            if membfunc['function'] == 'gaussian':
                MembershipLayer = _GaussianFuzzyLayer(membfunc['params'], membfunc['n_memb'])
            else:
                raise Exception(
                    'Membership function must be "gaussian".')

            fuzzyfication.append(MembershipLayer)

        self.fuzzyfication = fuzzyfication

    def reset_parameters(self):
        [layer.reset_parameters() for layer in self.fuzzyfication]

    def forward(self, input_):
        output = [Layer(input_[:, [i]])
                  for i, Layer in enumerate(self.fuzzyfication)]
        return output


class _GaussianFuzzyLayer(nn.Module):
    def __init__(self, params: dict, n_memb: int):
        super(_GaussianFuzzyLayer, self).__init__()
        self.params = params
        self.m = n_memb

        self._mu = torch.tensor([params['mu']['value']])
        self._sigma = torch.tensor([params['sigma']['value']])

        if params['mu']['trainable'] == True:
            self._mu = nn.Parameter(self._mu)

        if params['sigma']['trainable'] == True:
            self._sigma = nn.Parameter(self._sigma)

    @property
    def coeffs(self):
        return {'function': 'gaussian',
                'n_memb': self.m,
                'params': {'mu': {'value': self._mu.data.clone().flatten().tolist(),
                                  'trainable': isinstance(self._mu, nn.Parameter)},
                           'sigma': {'value': self._sigma.data.clone().flatten().tolist(),
                                     'trainable': isinstance(self._sigma, nn.Parameter)}
                           }
                }

    def reset_parameters(self):
        with torch.no_grad():
            self._mu[:] = torch.tensor([self.params['mu']['value']])
            self._sigma[:] = torch.tensor([self.params['sigma']['value']])

    def forward(self, input_):
        output = torch.exp(
            - torch.square(
                (input_.repeat(
                    1, self.m).reshape(-1, self.m) - self._mu)
                / self._sigma.square()
            )
        )
        return output

class _RuleLayer(nn.Module):
    def __init__(self):
        super(_RuleLayer, self).__init__()

    def forward(self, input_):
        n_in = len(input_)
        output = torch.ones(input_[0].shape).cuda()
        for i in range(n_in):
            output *= input_[i]

        return output


class _selectionLayer(nn.Module):
    def __init__(self, n_input, n_rules):
        super(_selectionLayer, self).__init__()
        self._s = nn.Parameter(torch.Tensor(n_input, n_rules))
        self.eps=torch.Tensor(1, n_rules).to("cuda:0")
        self.eps.fill_(0.001)
    def forward(self, input_):
        output = []
        for i in range(len(input_)):
            theta = 1/(1+torch.exp(-self._s[i]))
            top= input_[i]+self.eps
            a=(1-theta)*input_[i]
            b=a+theta
            bottom = b+self.eps
            output.append(top/bottom)
        return output,self._s
class _ConsequenceLayer(nn.Module):
    def __init__(self, n_input, n_rules):
        super(_ConsequenceLayer, self).__init__()
        self.n = n_input
        self.rules = n_rules

        # weights
        self._weight = nn.Parameter(torch.Tensor(self.n, n_rules))
        self._bias = nn.Parameter(torch.Tensor(1, n_rules))
        self.reset_parameters()

    @property
    def coeffs(self):
        return {'bias': self._bias,
                'weight': self._weight}

    @coeffs.setter
    def coeffs(self, new_coeffs: dict):
        assert type(
            new_coeffs) is dict, f'new coeffs should be dict filled with torch parameters, but {type(new_coeffs)} was given.'
        assert self._bias.shape == new_coeffs['bias'].shape and self._weight.shape == new_coeffs['weight'].shape, \
            f"New coeff 'bias' should be of shape {self._bias.shape}, but is instead {new_coeffs['bias'].shape} \n" \
            f"New coeff 'weight' should be of shape {self._weight.shape}, but is instead {new_coeffs['weight'].shape}"

        # transform to torch Parameter if any coeff is of type numpy array:
        if any(type(coeff) == np.ndarray for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(torch.from_numpy(
                new_coeffs[key]).float()) for key in new_coeffs}

        # transform to torch Parameter if any coeff is of type torch.Tensor:
        if any(type(coeff) == torch.Tensor for coeff in new_coeffs.values()):
            new_coeffs = {key: torch.nn.Parameter(
                new_coeffs[key].float()) for key in new_coeffs}

        self._bias = new_coeffs['bias']
        self._weight = new_coeffs['weight']

    def reset_parameters(self):
        with torch.no_grad():
            self._weight[:] = torch.rand(
                self.n, self.rules) - 0.5

            self._bias[:] = torch.rand(1, self.rules) - 0.5

    def forward(self, input_, wnorm,s):
        output = wnorm * (torch.matmul(input_, self._weight*s) + self._bias)
        return output
