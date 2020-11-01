from numpy import array, linspace, log
from samplers import ContrastiveDivergence as CD
from torch import cat, max, Tensor, zeros
from torch.utils.data import DataLoader

def logsumexp(tensor):
    '''Computes log(sum_i(exp(x_i))) for all elements in a torch tensor.
    The way of computing it without under- or overflows is through the
    log-sum-exp trick, namely computing
    log(1+exp(x)) = a + log(exp(-a) + exp(x-a))     with a = max(0, x)
    The function is adapted to be used in GPU if needed.

    Arguments:

        :param tensor: torch.Tensor
        :returns: torch.Tensor
    '''
    a = max(zeros(1).to(tensor.device), max(tensor))
    return a + (tensor - a).exp().sum().log()

def rbm_log_partition_function(rbm, batch_size, all_confs):
    '''Computes (via exact brute-force) the logarithm of the partition function
    of the Ising model defined by the weights of a Restricted Boltzmann Machine

    Arguments:

        :param rbm: Restricted Boltzmann Machine model
        :type rbm: :class:`ebm.models`
        :param batch_size: amount of samples used in every computation step
        :type batch_size: int
        :param all_confs: All possible configurations of the visible neurons
                          of the model
        :type all_confs: torch.Tensor

        :returns logZ: torch.Tensor with the logarithm of the partition function
    '''
    all_confs = DataLoader(all_confs.to(rbm.device), batch_size=batch_size)
    logsumexps = Tensor([]).to(rbm.device)
    for batch in all_confs:
        logsumexps = cat([logsumexps, logsumexp(rbm.free_energy(batch).neg())])
    logZ = logsumexp(logsumexps)
    return logZ

def rbm_log_partition_function_ais(rbm, batch_size, k=None,
                                   n_betas=1000, reverse=False):
    '''Approximates the logarithm of the partition function of the Ising model
    defined by the weights of a Restricted Boltzmann Machine using Annealed
    Importance Sampling (arXiv:1412.8566).
    Inspired by PyDeep http://pydeep.readthedocs.io

    Arguments:

        :param rbm: Restricted Boltzmann Machine model
        :type rbm: :class:`ebm.models`
        :param batch_size: amount of parallel runs employed
        :type batch_size: int
        :param k: Number of Gibbs steps per iteration
        :type k: int
        :param betas: Number of temperature steps
        :type k: int

        :returns logZ: torch.Tensor with the logarithm of the partition function
    '''
    betas = linspace(0, 1, n_betas)
    if reverse:
        betas = array(list(reversed(betas)))
    betas = Tensor(betas).to(rbm.device)
    sign = int(reverse)
    if k == None:
        try:
            k = rbm.sampler.k
        except AttributeError:
            k = 10
    sampler = CD(k)
    
    try:
        hbias = rbm.hbias.detach()
    except AttributeError:
        hbias = zeros(rbm.weights.shape[0]).to(rbm.device)
    try:
        vbias = rbm.vbias.detach()
    except AttributeError:
        vbias = zeros(rbm.weights.shape[1]).to(rbm.device)
    W = rbm.weights.detach()
    # Begin with a sample at infinite temperature
    v = sampler.get_v_from_h(zeros(batch_size, len(hbias)).to(rbm.device),
                             betas[0] * W, betas[0] * vbias)
    # Add to unnormalized probabilities
    log_factors = - (-1)**sign * rbm.free_energy(v * betas[0]).neg().detach()

    # Iterate on temperatures
    for beta in betas[1:-1]:
        log_factors += (-1)**sign * rbm.free_energy(v * beta).neg().detach()
        for _ in range(k):
            v = sampler.get_negative_sample(v,
                                            beta * W,
                                            beta * vbias,
                                            beta * hbias)
        log_factors -= (-1)**sign * rbm.free_energy(v * beta).neg().detach()
    # Add last element in the sequence
    log_factors += (-1)**sign * rbm.free_energy(v * betas[-1]).neg().detach()

    logZ_estimate = logsumexp(log_factors) - log(batch_size)
    logZ = log(2) * (len(hbias) + len(vbias)) + logZ_estimate
    return logZ
