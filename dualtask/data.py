#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:05:34 2019

@author: molano
"""
import numpy as np
import matplotlib.pylab as plt


def get_inputs_outputs(n_batch, n_time, n_bits, gng_time):
    # inputs mat
    inputs = np.zeros([n_batch, n_time, n_bits])
    # build dpa structure
    dpa_stim1 = np.arange((n_bits-2)/2)
    stim1_seq, choice1 = get_stims(dpa_stim1, n_batch)
    dpa_stim2 = np.arange((n_bits-2)/2, (n_bits-2))
    stim2_seq, choice2 = get_stims(dpa_stim2, n_batch)
    # ground truth dpa
    gt_dpa = choice1 == choice2

    # build go-noGo task if required
    if gng_time != 0:
        gng_stim = np.arange((n_bits-2), n_bits)
        gng_stim_seq, gt_gng = get_stims(gng_stim, n_batch)

    for ind_btch in range(n_batch):
        inputs[ind_btch, 0, stim1_seq[ind_btch]] = 1
        inputs[ind_btch, n_time-2, stim2_seq[ind_btch]] = 1
        if gng_time != 0:
            inputs[ind_btch, gng_time-1, gng_stim_seq[ind_btch]] = 1

    # output (note that n_bits could actually be 1 here because we just
    # need one decision. I kept it as it is for the flipFlop task
    # to avoid issues in other parts of the algorithm)
    outputs = np.zeros([n_batch, n_time, n_bits])
    outputs[gt_dpa == 1, n_time-1, 0] = 1
    outputs[gt_gng == 1, gng_time, 0] = 1
    return {'inputs': inputs, 'output': outputs}


def get_stims(stim, n_batch):
    choice = np.random.choice(stim.shape[0], size=(n_batch, ))
    stim_seq = stim[choice].astype(int)
    return stim_seq, choice


if __name__ == '__main__':
    plt.close('all')
    n_batch = 4
    n_time = 8
    n_bits = 6
    gng_time = 3
    example_trials = get_inputs_outputs(n_batch, n_time, n_bits, gng_time)
#    print(example_trials['inputs'][0, :, :].T)
#    print('----')
#    print(example_trials['inputs'][1, :, :].T)
#    print('----')
#    print(example_trials['inputs'][2, :, :].T)
#    print('----')
#    print(example_trials['inputs'][3, :, :].T)
#    print('----')
#    print(example_trials['output'].shape)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.squeeze(example_trials['inputs'][0, :, :].T), aspect='auto')
    plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(example_trials['output'][0, :, 0]))
