# Collection of energy-based ML models
#
# Author: Alejandro Pozas-Kerstjens
# Requires: copy for variable copies
#           numpy for numerics
#           pytorch as ML framework
#           tqdm for progress bar
# Last modified: Jul, 2019

import torch
from copy import deepcopy
from .optimizers import outer_product
from torch.nn import Module, Parameter
from torch.nn.functional import linear, softplus
from tqdm import tqdm


class RBM(Module):

    def __init__(self, n_visible=100, n_hidden=50, sampler=None, optimizer=None,
                 device=None, weights=None, hbias=None, vbias=None):
        '''Constructor for the class.

        Arguments:

            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param verbose: Optional parameter to set verbosity mode
            :type verbose: int
            :param weights: Optional parameter to specify the weights of the RBM
            :type weights: torch.nn.Parameter
            :param hbias: Optional parameter to specify the hidden biases of
                          the RBM
            :type hbias: torch.nn.Parameter
            :param vbias: Optional parameter to specify the visibile biases of
                          the RBM
            :type vbias: torch.nn.Parameter
        '''

        super(RBM, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')

        if weights is not None:
            self.weights = Parameter(weights.to(self.device))
        else:
            self.weights = Parameter(0.01 * torch.randn(n_hidden,
                                                        n_visible
                                                        ).to(self.device))

        if hbias is not None:
            self.hbias = Parameter(hbias.to(self.device))
        else:
            self.hbias = Parameter(torch.zeros(n_hidden).to(self.device))

        if vbias is not None:
            self.vbias = Parameter(vbias.to(self.device))
        else:
            self.vbias = Parameter(torch.zeros(n_visible).to(self.device))
        
        for param in self.parameters():
            param.requires_grad = False

        if optimizer is None:
            raise Exception('You must provide an appropriate optimizer')
        self.optimizer = optimizer

        if sampler is None:
            raise Exception('You must provide an appropriate sampler')
        self.sampler = sampler

    def free_energy(self, v):
        '''Computes the free energy for a given state of the visible layer.

        Arguments:

            :param v: The state of the visible layer of the RBM
            :type v: torch.Tensor

            :returns: torch.Tensor
        '''
        vbias_term = v.mv(self.vbias)
        wx_b = linear(v, self.weights, self.hbias)
        hidden_term = softplus(wx_b).sum(1)
        return (-hidden_term - vbias_term)

    def train(self, input_data):
        '''Trains the RBM.

        Arguments:

            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
        '''
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):
            sample_data = batch.float()
            # Sampling from the model to compute updates
            # Get positive phase from the data
            vpos = sample_data
            # Get negative phase from the chains
            vneg = self.sampler.get_negative_sample(vpos, self.weights,
                                                    self.vbias, self.hbias)

            # Weight updates. Includes momentum and weight decay
            W_update, vbias_update, hbias_update = \
                            self.optimizer.get_updates(vpos, vneg, self.weights,
                                                        self.vbias, self.hbias)

            self.weights += W_update
            self.hbias   += hbias_update
            self.vbias   += vbias_update

        self.optimizer.epoch += 1


class DBN(object):
    def __init__(self, n_visible=6, hidden_layer_sizes=[3, 3],
                 sample_copies=1, sampler=None, optimizer=None,
                 continuous_output=False, device=None):
        '''Constructor for the class.
        
        Arguments:
        
            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param hidden_layer_sizes: The number nodes in each of
                                       the hidden layers
            :type hidden_layer_sizes: list of int
            :param sample_copies: How many samples from a hidden layer are
                                  drawn to train the next layer
            :type sample_copies: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param continuous_output: Optional parameter to indicate whether
                                      the visible layer is continuous-valued.
            :type continuous_output: bool
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
        '''
        self.sample_copies     = sample_copies
        self.continuous_output = continuous_output
        
        self.gen_layers = []
        self.inference_layers = []
        self.n_layers   = len(hidden_layer_sizes)
        
        assert self.n_layers > 0, 'You must specify at least one hidden layer'

        if device is None:
            device = torch.device('cpu')
        self.device = device

        if optimizer is None:
            raise Exception('You must provide an appropriate optimizer')
        self.optimizer = optimizer

        if sampler is None:
            raise Exception('You must provide an appropriate sampler')
        self.sampler = sampler

        # Construct DBN out of RBMs
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = hidden_layer_sizes[i - 1]

            gen_layer = RBM(n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            sampler=sampler,
                            optimizer=optimizer,
                            device=device)
            inf_layer = RBM(n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            sampler=sampler,
                            optimizer=optimizer,
                            device=device)
            self.gen_layers.append(gen_layer)
            self.inference_layers.append(inf_layer)
            
    def pretrain(self, input_data, epochs=15, batch_size=10, test=None):
        '''Pre-trains the DBN as a sequence of RBMs.
        
        Arguments:
        
            :param input_data: Dataset
            :type input_data: torch.utils.data.DataLoader
            :param lr: Learning rate
            :type lr: float
            :param weight_decay: Weight decay parameter, to prevent overfitting
            :type weight_decay: float
            :param momentum: Momentum parameter, for improved learning
            :type momentum: float
            :param epochs: Number of epochs of training
            :type epochs: int
            :param batch_size: Batch size
            :type batch_size: int
            :param test: Optional parameter, dataset to validate model
            :type test: torch.utils.data.DataLoader
        '''
        for i in range(self.n_layers):
            print('#########################################')
            print('#        Pre-training layers {}-{}        #'.format(i, i+1))
            print('#########################################')
            if i == 0:
                layer_input = input_data.to(self.device)
                # In case the visible layer is continuous, we have to let the
                # sampler know
                self.sampler.continuous_output = self.continuous_output
            else:
                self.sampler.continuous_output = False
                total_layer_input = []
                for _ in range(self.sample_copies):
                    sample = self.sampler.get_h_from_v(layer_input,
                                                   self.gen_layers[i-1].weights,
                                                   self.gen_layers[i-1].hbias)
                    total_layer_input.append(sample.data)
                layer_input = torch.cat(total_layer_input, 0)
                if test is not None:
                    test = self.sampler.get_h_from_v(test,
                                                   self.gen_layers[i-1].weights,
                                                   self.gen_layers[i-1].hbias)
            
            self.sampler.first_call = True
            self.optimizer.first_call = True
            rbm = self.inference_layers[i]
            for epoch in range(epochs):
                layer_loader = torch.utils.data.DataLoader(layer_input,
                                                           batch_size,
                                                           True)
                rbm.train(layer_loader)
                if test is not None:
                    validation = layer_input[:test.size(0)]
                    val_fe  = rbm.free_energy(validation).mean(0)
                    test_fe = rbm.free_energy(test).mean(0)
                    gap = val_fe - test_fe
                    print('Gap: ' + str(gap.item()))

    def finetune(self, input_data, lr=0.1, epochs=100, batch_size=10):
        '''Fine-tuning is done according to the up-down algorithm developed in
        Hinton et al., A fast learning algorithm for deep belief nets.
        Neural Computation 18, 1527-1554 (2006)
           
        Arguments:
        
            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
            :param lr: Learning rate
            :type lr: float
            :param epochs: Number of epochs of training
            :type epochs: int
            :param batch_size: Batch size
            :type batch_size: int
        '''
        print('#########################################')
        print('#        Fine-tuning full model         #')
        print('#########################################')
        
        # First of all, copy the weights to the generative layers.
        for i, inf in enumerate(self.inference_layers):
            self.gen_layers[i].load_state_dict(deepcopy(inf.state_dict()))
        
        input_loader = torch.utils.data.DataLoader(input_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        top_RBM = self.gen_layers[-1]
        for epoch in range(epochs):
            for idx, batch in enumerate(tqdm(input_loader,
                                              desc='Epoch ' + str(epoch + 1))):
                sample_data = batch.float().to(self.device)
                # Perform a bottom-up pass to get wake positive phase
                # probabilities and samples, using the inference parameters.
                # We begin with the train data as samples of the visible layer
                wakepos_samples = [sample_data]
                self.sampler.internal_sampling = True
                for i, rbm in enumerate(self.inference_layers[:-1]):
                    layer_input = wakepos_samples[-1]
                    sample = self.sampler.get_h_from_v(layer_input,
                                                       rbm.weights,
                                                       rbm.hbias)
                    wakepos_samples.append(sample)
                
                sample = self.sampler.get_h_from_v(wakepos_samples[-1],
                                                   top_RBM.weights,
                                                   top_RBM.hbias)
                wakepos_samples.append(sample)
                self.sampler.internal_sampling = False
                # We use these parameters to get the positive phase of the top
                # layer, which is trained as a standard RBM
                pos_W_top = outer_product(wakepos_samples[-1],
                                          wakepos_samples[-2])
                # Perform Gibbs sampling iterations in the top RBM
                self.sampler.continuous_output = False
                if (idx == 0) and (epoch == 0):
                    self.sampler.first_call = True
                v_sample = self.sampler.get_v_sample(wakepos_samples[-2],
                                                     top_RBM.weights,
                                                     top_RBM.vbias,
                                                     top_RBM.hbias)
                h_sample = self.sampler.get_h_from_v(v_sample,
                                                     top_RBM.weights,
                                                     top_RBM.hbias)
                # Take negative phase statistics on the top RBM
                neg_W_top = outer_product(h_sample, v_sample)
                # Beginning at the end of the sampling, perform a top-down
                # generative pass to obtain the sleep positive phase samples
                sleeppos_samples = [h_sample]
                for i, rbm in reversed(list(enumerate(self.gen_layers))):
                    layer_input = sleeppos_samples[-1]
                    if i == 0:
                        self.sampler.continuous_output = self.continuous_output
                    sleeppos_sample = self.sampler.get_v_from_h(layer_input,
                                                                rbm.weights,
                                                                rbm.vbias)
                    sleeppos_samples.append(sleeppos_sample)
                # Go back to normal order, where the first element corresponds
                # to the visible layer
                sleeppos_samples = list(reversed(sleeppos_samples))
                # Predictions on the current states of the layers, for
                # the negative contributions of the wake and sleep phases.
                # Note we use probabilities instead of samples, as appears in
                # the original algorithm
                sleepneg_samples = [None]  # For equalling lengths. Unimportant
                wakeneg_samples  = []
                self.sampler.internal_sampling  = False
                self.sampler.hidden_activations = True
                for i in range(self.n_layers):
                    if i == 0:
                        self.sampler.continuous_output = self.continuous_output
                    else:
                        self.sampler.continuous_output = False
                    sleepneg_sample = (
                               self.sampler.get_h_from_v(sleeppos_samples[i],
                                               self.inference_layers[i].weights,
                                               self.inference_layers[i].hbias)
                                       )
                    wakeneg_sample = (
                                self.sampler.get_v_from_h(wakepos_samples[i+1],
                                                     self.gen_layers[i].weights,
                                                     self.gen_layers[i].vbias)
                                      )
                    sleepneg_samples.append(sleepneg_sample)
                    wakeneg_samples.append(wakeneg_sample)
                # Updates to generative parameters. The last layer still acts
                # as a standard RBM and we update it separately
                for i, rbm in enumerate(self.gen_layers[:-1]):
                    wakediff_i = wakepos_samples[i] - wakeneg_samples[i]
                    deltaW     = outer_product(wakepos_samples[i+1],
                                               wakediff_i).mean(0)
                    deltav     = wakediff_i.mean(0)
                    rbm.weights.data += lr * deltaW.data
                    rbm.vbias.data   += lr * deltav.data
                    # The lack of update to hidden biases is because the
                    # generation weights are only used in top-down propagation
                # Updates to top RBM parameters
                deltaW = (pos_W_top - neg_W_top)
                deltav = (wakepos_samples[-2] - sleeppos_samples[-2])
                deltah = (wakepos_samples[-1] - sleeppos_samples[-1])
                top_RBM.weights.data += lr * deltaW.mean(0)
                top_RBM.vbias.data   += lr * deltav.mean(0)
                top_RBM.hbias.data   += lr * deltah.mean(0)
                # Updates to inference parameters
                for i, rbm in enumerate(self.inference_layers):
                    sleepdiff_i = sleeppos_samples[i+1] - sleepneg_samples[i+1]
                    deltaW      = outer_product(sleepdiff_i,
                                                sleeppos_samples[i]).mean(0)
                    deltah      = sleepdiff_i.mean(0)
                    rbm.weights.data += lr * deltaW
                    rbm.hbias.data   += lr * deltah
                    # As above, the lack of visible bias update is because the
                    # inference weights are only used in bottom-up propagation

    def generate(self, k=None):
        '''Generates samples from the DBN.
           
        Arguments:
        
            :param k: In case of (Persistent) Contrastive Divergence sampling,
                      the number of iteration steps before propagating the
                      sample down the network.
            :type k: int
            
            :returns: torch.Tensor -- a sample from the visible layer 
        '''
        if k is not None:
            self.sampler.k = k
        rbm    = self.gen_layers[-1]
        sample = torch.zeros(rbm.vbias.size()).to(self.device)
        self.sampler.continuous_output = False
        sample = self.sampler.get_v_sample(sample, rbm.weights,
                                           rbm.vbias, rbm.hbias)
        for i, rbm_layer in reversed(list(enumerate(self.gen_layers[:-1]))):
            if i == 0:
                self.sampler.continuous_output = self.continuous_output
            sample = self.sampler.get_v_from_h(sample,
                                               rbm_layer.weights,
                                               rbm_layer.vbias)
        return sample

    def save_model(self, filename):
        '''Saves the parameters for all layers in the network.
           
        Arguments:
        
            :param filename: Filename
            :type filename: str
        '''
        dicts = []
        for layer in self.gen_layers + self.inference_layers:
            dicts.append(layer.state_dict())
        torch.save(dicts, filename)

    def load_model(self, filename):
        '''Loads a DBN stored in disk.
           
        Arguments:
        
            :param filename: Filename
            :type filename: str
        '''
        dicts = torch.load(filename)
        for i, layer in enumerate(self.gen_layers + self.inference_layers):
            layer.load_state_dict(dicts[i])

class BM(Module):

    def __init__(self, n_nodes=100, sampler=None, optimizer=None,
                 device=None, weights=None, bias=None):
        '''Constructor for the class.
        Arguments:
            :param n_nodes: The number of nodes
            :type n_nodes: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param weights: Optional parameter to specify the weights of the BM
            :type weights: torch.nn.Parameter
            :param bias: Optional parameter to specify the biases of the BM
            :type bias: torch.nn.Parameter
        '''

        super(BM, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        
        if weights is not None:
            self.weights = Parameter(weights.to(self.device))
        else:
            rnd = 0.01 * np.random.randn(n_nodes, n_nodes)
            rnd = rnd + rnd.T
            np.fill_diagonal(rnd, 0)
            self.weights = Parameter(torch.from_numpy(rnd).type(torch.float).to(self.device))

        if bias is not None:
            self.bias = Parameter(bias.to(self.device))
        else:
            self.bias = Parameter(torch.Tensor(
                                               torch.zeros(n_nodes)
                                               ).to(self.device))
        self.register_parameter('bias', self.bias)

        for param in self.parameters():
            param.requires_grad = False

        if optimizer is None:
            raise Exception('You must provide an appropriate optimizer')
        self.optimizer = optimizer

        if sampler is None:
            raise Exception('You must provide an appropriate sampler')
        self.sampler = sampler

    def energy(self, v):
        '''Computes the energy for a given state of the nodes.
        Arguments:
            :param v: The state of the visible layer of the RBM
            :type v: torch.Tensor
            :returns: torch.Tensor
        '''
        bias_term = v.mv(self.bias)
        xwx = torch.einsum('bi,ij,bj->b', (v, self.weights, v))
        return (-xwx - bias_term)

    def train(self, input_data):
        '''Trains the RBM.
        Arguments:
            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
            :param lr: Learning rate
            :type lr: float
            :param weight_decay: Weight decay parameter, to prevent overfitting
            :type weight_decay: float
            :param momentum: Momentum parameter, for improved learning
            :type momentum: float
            :param epoch: Optional parameter to show epoch in screen
            :type epoch: int
        '''
        error_ = []
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):
            sample_data = batch.float()
            # Sampling from the model to compute updates
            # Get positive phase from the data
            vpos = sample_data
            # Get negative phase from the chains
            vneg = self.sampler.get_negative_phase(vpos, self.weights,
                                                   self.bias).to(self.device)

            # Weight updates. Includes momentum and weight decay
            W_update, bias_update = \
                             self.optimizer.get_updates(vpos, vneg,
                                                        self.weights, self.bias)

            self.weights.data    += W_update.data
            self.bias.data += bias_update.data
            
        self.optimizer.epoch += 1
