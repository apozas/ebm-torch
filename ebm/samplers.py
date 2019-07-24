# Collection of samplers for Energy-based models in pytorch
#
# Author: Alejandro Pozas-Kerstjens
# Requires: pytorch as ML framework
# Last modified: Jul, 2019

from torch import cat, einsum, rand, rand_like, sigmoid
from torch.nn.functional import dropout, linear

class Sampler(object):

    def __init__(self):
        """Constructor for the class.
        """

    def get_negative_sample(self, v0, weights, vbias, hbias):
        """Samples the visible layer from p(v)=sum_h p(v,h)

        Arguments:

            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param weights: Weights connecting the visible and hidden layers
            :type weights: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      visible nodes
        """

    def get_h_from_v(self, v0, weights, hbias):
        """Samples the hidden layer from the conditional distribution p(h|v)

        Arguments:

            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param weights: Weights connecting the visible and hidden layers
            :type weights: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      hidden nodes
        """

    def get_v_from_h(self, h0, weights, vbias):
        """Samples the visible layer from the conditional distribution p(v|h)

        Arguments:

            :param h0: Initial configuration of the hidden nodes
            :type h0: torch.Tensor
            :param weights: Weights connecting the visible and hidden layers
            :type weights: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      visible nodes
        """


class ContrastiveDivergence(Sampler):

    def __init__(self, k, dropout=0, continuous_output=False):
        """Constructor for the class.

        Arguments:

            :param k: The number of iterations in CD-k
            :type k: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
            :param continuous_output: Optional parameter to output visible
                                      activations instead of samples (for
                                      continuous-valued outputs)
            :type continuous_output: bool
        """
        super(ContrastiveDivergence, self).__init__()
        assert k > 0, 'You should specify a number of Gibbs steps > 0'
        self.k = k
        assert (dropout >= 0) and (dropout <= 1), ('The dropout rate' +
                                                   ' should be in [0, 1]')
        self.dropout = dropout
        self.continuous_output = continuous_output

    def get_negative_sample(self, v0, weights, vbias, hbias):
        v = None
        for _ in range(self.k):
            if v is None:
                h = self.get_h_from_v(v0, weights, hbias)
            else:
                h = self.get_h_from_v(v, weights, hbias)
            v = self.get_v_from_h(h, weights, vbias)
        return v

    def get_h_from_v(self, v, weights, hbias):
        h_probs = self._propup(v, weights, hbias)
        return h_probs.bernoulli()

    def get_v_from_h(self, h, weights, vbias):
        v_probs = self._propdown(h, weights, vbias)
        v_sample = v_probs.bernoulli()
        return v_probs if self.continuous_output else v_sample

    def _propdown(self, h, weights, vbias):
        pre_sigmoid_activation = linear(dropout(h, self.dropout),
                                        weights.t(), vbias)
        return sigmoid(pre_sigmoid_activation)

    def _propup(self, v, weights, hbias):
        pre_sigmoid_activation = linear(dropout(v, self.dropout),
                                        weights, hbias)
        return sigmoid(pre_sigmoid_activation)

class PersistentContrastiveDivergence(ContrastiveDivergence):

    def __init__(self, k, n_chains=0, dropout=0, continuous_output=False):
        """Constructor for the class.

        Arguments:

            :param k: The number of iterations in PCD-k
            :type k: int
            :param n_chains: Number of parallel chains (negative particles)
                             to be run
            :type n_chains: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
            :param continuous_output: Optional parameter to output visible
                                      activations instead of samples (for
                                      continuous-valued outputs)
            :type continuous_output: bool
        """
        super().__init__(k, dropout, continuous_output)
        self.first_call = True
        self.n_chains = n_chains

    def get_negative_sample(self, v0, weights, vbias, hbias):
        if self.first_call:
            if self.n_chains <= 0:
                self.markov_chains = rand_like(v0)
            else:
                self.markov_chains = rand((self.n_chains,) + v0.size()[1:]
                                          ).to(v0.device)
            self.first_call = False
        for _ in range(self.k):
            h = self.get_h_from_v(self.markov_chains, weights, hbias)
            v = self.get_v_from_h(h, weights, vbias)
            self.markov_chains = v
        return v

class ParallelTempering(PersistentContrastiveDivergence):

    def __init__(self, k, n_chains=0, betas=[1],
                 dropout=0, continuous_output=False):
        """Constructor for the class.

        Arguments:

            :param k: The number of Gibbs steps
            :type k: int
            :param n_chains: Number of parallel chains (negative particles)
                             to be run
            :type n_chains: int
            :param betas: List of inverese temperatures of the copies of
                          negative particles. If int, number of inverse
                          temperatures equispaced between 0 and 1 of the copies
                          of the negative particles
            :type betas: list or 1-d array or int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
            :param continuous_output: Optional parameter to output visible
                                      activations instead of samples (for
                                      continuous-valued outputs)
            :type continuous_output: bool
        """
        super().__init__(k, n_chains, dropout, continuous_output)
        self.first_call = True
        if type(betas) == int:
            self.betas = np.linspace(0, 1, num=betas)[::-1]
        else:
            if betas[0] < betas[1]:
                self.betas = betas[::-1]
            else:
                self.betas = betas

    def get_negative_sample(self, v0, weights, vbias, hbias):
        if self.first_call:
            if self.n_chains <= 0:
                self.markov_chains = rand(v0.size() + (len(self.betas,))
                                          ).to(v0.device)
            else:
                self.markov_chains = rand((self.n_chains,) + v0.size()[1:]
                                           + (len(self.betas),)).to(v0.device)
            self.first_call = False
        for _ in range(self.k):
            h = cat([self.get_h_from_v(
                                      self.markov_chains[:, :, i].squeeze(),
                                      weights * beta, hbias * beta).unsqueeze(2)
                           for i, beta in enumerate(self.betas)], 2)
            v = cat([self.get_v_from_h(h[:, :, i].squeeze(),
                                      weights * beta, vbias * beta).unsqueeze(2)
                          for i, beta in enumerate(self.betas)], 2)
            v = self._swap_chains(v, h, weights, vbias, hbias)
            self.markov_chains.data = v
        return self.markov_chains[:, :, 0].squeeze()

    def _swap_chains(self, v, h, weights, vbias, hbias):
        energies = self._energy(v, h, weights, vbias, hbias)
        for i in range(len(self.betas) - 1):
            deltas = (self.betas[i] - self.betas[i+1])                     \
                     * (energies[:, i] - energies[:, i+1]).squeeze()
            swap = deltas.exp() > rand_like(deltas)
            for b, perm in enumerate(swap):
                if perm:
                    v[b, :, [i, i+1]]     = v[b, :, [i+1, i]]
                    energies[b, [i, i+1]] = energies[b, [i+1, i]]
        return v

    def _energy(self, v, h, weights, vbias, hbias):
        vWh = einsum('bit,ji,bjt->bt', (v, weights, h))
        vb = einsum('bit,i->bt', (v, vbias))
        hc = einsum('bit,i->bt', (h, hbias))
        return -vWh - vb - hc
