# Example of usage: Deep Belief Network
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           imageio for output export
# Last modified: Jun, 2018

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from ebm.models import DBN
from ebm.optimizers import SGD
from ebm.samplers import PersistentContrastiveDivergence
from torchvision import datasets

#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidden_layers   = [30, 30]    # Number of nodes on each hidden layer
pretrain_lr     = 1e-2        # Learning rate for pre-training
weight_decay    = 1e-4        # Weight decay for pre-training
momentum        = 0.95        # Momentum for pre-training
pretrain_epochs = 20          # Pre-training epochs
k               = 5           # Steps of contrastive divergence in pre-training
finetune_lr     = 1e-4        # Learning rate for fine-tuning
finetune_epochs = 30          # Finetuning epochs
batch_size      = 20          # Batch size
gpu             = False       # Use GPU
continuous      = True        # Whether we want continuous outputs or not
sample_copies   = 5           # Number of samples taken from the hidden
                              # representation of each datapoint

#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')

data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.float)
                      
test = datasets.MNIST('mnist',
                      train=False).test_data.type(torch.float)

data = (data.view((-1, 784)) / 255).to(device)
test = (test.view((-1, 784)) / 255).to(device)

vis  = len(data[0])

# -----------------------------------------------------------------------------
# Construct DBN
# -----------------------------------------------------------------------------
pre_trained = os.path.isfile('DBN.h5')

sampler = PersistentContrastiveDivergence(k=k, continuous_output=continuous)
optimizer = SGD(learning_rate=pretrain_lr,
                momentum=momentum,
                weight_decay=weight_decay)
dbn = DBN(n_visible=vis,
          hidden_layer_sizes=hidden_layers,
          sample_copies=sample_copies,
          sampler=sampler,
          optimizer=optimizer,
          continuous_output=continuous,
          device=device)
if pre_trained:
    dbn.load_model('DBN.h5')
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
if not pre_trained:
    dbn.pretrain(input_data=data,
                 epochs=pretrain_epochs,
                 batch_size=batch_size,
                 test=test)
    dbn.finetune(input_data=data,
                 lr=finetune_lr,
                 epochs=finetune_epochs,
                 batch_size=batch_size)

    dbn.save_model('DBN.h5')

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print('#########################################')
print('#          Generating samples           #')
print('#########################################')
top_RBM = dbn.gen_layers[-1]
plt.figure(figsize=(20, 10))
zero = torch.zeros(25, len(top_RBM.vbias)).to(device)
images = [np.zeros((5 * 28, 5 * 28))]
for i in range(200):
    sampler.continuous_output = False
    zero = sampler.get_h_from_v(zero, top_RBM.weights, top_RBM.hbias)
    zero = sampler.get_v_from_h(zero, top_RBM.weights, top_RBM.vbias)
    sample = zero
    for gen_layer in reversed(dbn.gen_layers[1:-1]):
        sample = sampler.get_v_from_h(sample, gen_layer.weights, gen_layer.vbias)
    sampler.continuous_output = continuous
    sample = sampler.get_v_from_h(sample,
                                  dbn.gen_layers[0].weights,
                                  dbn.gen_layers[0].vbias)
    datas = sample.data.cpu().numpy().reshape((25, 28, 28))
    image = np.zeros((5 * 28, 5 * 28))
    for k in range(5):
        for l in range(5):
            image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
    images.append(image)
imageio.mimsave('DBN_sample.gif', images, duration=0.1)
