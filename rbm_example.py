# Example of usage: Restricted Boltzmann Machine with continuous-valued outputs
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
# Last modified: Jun, 2018

import imageio
import numpy as np
import os
import torch
from ebm.optimizers import Adam
from ebm.samplers import ParallelTempering
from ebm.models import RBM
from torchvision import datasets

#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd           = 300          # Number of nodes in the hidden layer
learning_rate  = 1e-3         # Learning rate
epochs         = 20           # Training epochs
k              = 1            # Steps of Contrastive Divergence
k_reconstruct  = 2000         # Steps of iteration during generation
batch_size     = 32           # Batch size
model_dir      = 'RBM.h5'     # Directory for saving last parameters learned
gpu            = True         # Use of GPU

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

vis = len(data[0])

# According to Hinton this initialization of the visible biases should be
# fine, but some biases diverge in the case of MNIST.
# Actually, this initialization is the inverse of the sigmoid. This is, it
# is the inverse of p = sigm(vbias), so it can be expected that during
# training the weights are close to zero and change little
vbias = torch.log(data.mean(0)/(1 - data.mean(0))).clamp(-20, 20)

# -----------------------------------------------------------------------------
# Construct RBM
# -----------------------------------------------------------------------------
sampler = ParallelTempering(k=k,
                            n_chains=100,
                            betas=[0, 0.25, 0.5, 0.75, 1],
                            continuous_output=True)
optimizer = Adam(learning_rate)
rbm = RBM(n_visible=vis,
          n_hidden=hidd,
          sampler=sampler,
          optimizer=optimizer,
          device=device,
          vbias=vbias)
pre_trained = os.path.isfile(model_dir)
if pre_trained:
    rbm.load_state_dict(torch.load(model_dir))

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
if not pre_trained:
    validation = data[:10000]
    for _ in range(epochs):
        train_loader = torch.utils.data.DataLoader(data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        rbm.train(train_loader)
        # A good measure of well-fitting is the free energy difference
        # between some known and unknown instances.
        gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)
        print('Gap = {}'.format(gap.item()))

    torch.save(rbm.state_dict(), model_dir)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print('Reconstructing images')
zero = torch.zeros(25, 784).to(device)
images = [zero.cpu().numpy().reshape((5 * 28, 5 * 28))]
for i in range(k_reconstruct):
    zero = sampler.get_h_from_v(zero, rbm.W, rbm.hbias)
    zero = sampler.get_v_from_h(zero, rbm.W, rbm.vbias)
    if i % 3 == 0:
        datas = zero.data.cpu().numpy().reshape((25, 28, 28))
        image = np.zeros((5 * 28, 5 * 28))
        for k in range(5):
            for l in range(5):
                image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
        images.append(image)
imageio.mimsave('RBM_sample.gif', images, duration=0.1)
