'''
run_flipflop.py
Written using Python 2.7.12
@ Matt Golub, October 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
PATH_TO_FIXED_POINT_FINDER = '../'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from DualTask import DualTask
from FixedPointFinder import FixedPointFinder
from FixedPoints import FixedPoints

# *****************************************************************************
# STEP 1: Train an RNN to solve the dual task *********************************
# *****************************************************************************

# Hyperparameters for AdaptiveLearningRate
alr_hps = {'initial_rate': 0.1}

# Hyperparameters for FlipFlop
# See FlipFlop.py for detailed descriptions.
hps = {
    'rnn_type': 'vanilla',
    'n_hidden': 256,
    'min_loss': 1e-6,  # 1e-4
    'min_learning_rate': 1e-10,  # 1e-5
    'max_n_epochs': 10000,
    'do_restart_run': False,
    'log_dir': './logs/',
    'data_hps': {
        'n_batch': 2048,
        'n_time': 20,
        'n_bits': 6,
        'noise': 0.05,
        'gng_time': 0,
        'lamb': 0,
        'delay_max': 17},
    'alr_hps': alr_hps
    }

dt = DualTask(**hps)
dt.train()

# Get example state trajectories from the network
# Visualize inputs, outputs, and RNN predictions from example trials
example_trials = dt.generate_dualtask_trials()
f = dt.plot_trials(example_trials)
# f.canvas.draw()
# *****************************************************************************
# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN ****
# *****************************************************************************
'''Initial states are sampled from states observed during realistic behavior
of the network. Because a well-trained network transitions instantaneously
from one stable state to another, observed networks states spend little if any
time near the unstable fixed points. In order to identify ALL fixed points,
noise must be added to the initial states before handing them to the fixed
point finder.'''
NOISE_SCALE = 0.5  # Standard deviation of noise added to initial states
N_INITS = 1024  # The number of initial states to provide

n_bits = dt.hps.data_hps['n_bits']
is_lstm = dt.hps.rnn_type == 'lstm'

'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
descriptions of available hyperparameters.'''
fpf_hps = {}

# Setup the fixed point finder
fpf = FixedPointFinder(dt.rnn_cell,
                       dt.session,
                       **fpf_hps)

# Study the system in the absence of input pulses (e.g., all inputs are 0)
inputs = np.zeros([1, n_bits])

'''Draw random, noise corrupted samples of those state trajectories
to use as initial states for the fixed point optimizations.'''
example_predictions = dt.predict(example_trials,
                                 do_predict_full_LSTM_state=is_lstm)
initial_states = fpf.sample_states(example_predictions['state'],
                                   n_inits=N_INITS,
                                   rng=dt.rng,
                                   noise_scale=NOISE_SCALE)
# plot population activity
f = plt.figure()
num_plots = 5
for ind_pl in range(num_plots):
    plt.subplot(num_plots, 1, ind_pl+1)
    aux = np.squeeze(example_predictions['state'][ind_pl, :, :].T)
    maximo = np.max(np.abs(aux), axis=1).reshape((dt.hps.n_hidden, 1))
#    print(aux)
    aux = aux/maximo
#    print(maximo)
    plt.imshow(aux, aspect='auto')
    if ind_pl == num_plots-1:
        plt.xlabel('time (a.u.)')
        plt.ylabel('neurons')

f = plt.figure()
num_plots = 5
for ind_pl in range(num_plots):
    plt.subplot(num_plots, 1, ind_pl+1)
    aux = np.squeeze(example_predictions['state'][ind_pl, :, :].T)
    maximo = np.max(np.abs(aux), axis=1).reshape((dt.hps.n_hidden, 1))
    plt.plot(aux.T)
    if ind_pl == num_plots-1:
        plt.xlabel('time (a.u.)')
        plt.ylabel('neurons')

# Run the fixed point finder
unique_fps, _ = fpf.find_fixed_points(initial_states, inputs)

# Visualize identified fixed points with overlaid RNN state trajectories
# All visualized in the 3D PCA space fit the the example RNN states.
# example_trials['stim_conf'] specifies the color of the trace
# so I just lower the intensity of the colors that are too bright
# suma = np.sum(example_trials['stim_conf'], axis=1).\
#    reshape((example_trials['stim_conf'].shape[0], 1)) + 0.000001
# example_trials['stim_conf'] =\
#     example_trials['stim_conf']/suma
# colors based on S1/S2
colors = np.zeros_like(example_trials['stim_conf'])
colors[:, 0] = example_trials['stim_conf'][:, 0]
f = unique_fps.plot(example_predictions['state'],
                    stim_config=colors,
                    plot_batch_idx=range(128),
                    gng_time=dt.hps.data_hps['gng_time'], title='S1/S2')
# colors based on S3/s4
colors = np.zeros_like(example_trials['stim_conf'])
colors[:, 0] = example_trials['stim_conf'][:, 1]
f = unique_fps.plot(example_predictions['state'],
                    stim_config=colors,
                    plot_batch_idx=range(128),
                    gng_time=dt.hps.data_hps['gng_time'], title='S3/S4')
# colors based on S5/s6
colors = np.zeros_like(example_trials['stim_conf'])
colors[:, 0] = example_trials['stim_conf'][:, 2]
f = unique_fps.plot(example_predictions['state'],
                    stim_config=colors,
                    plot_batch_idx=range(128),
                    gng_time=dt.hps.data_hps['gng_time'], title='S5/S6')

# colors based on final GO/NOGO
colors = np.zeros_like(example_trials['stim_conf'])
colors[:, 0] = example_trials['output'][:, -1, 0]
f = unique_fps.plot(example_predictions['state'],
                    stim_config=colors,
                    plot_batch_idx=range(128),
                    gng_time=dt.hps.data_hps['gng_time'], block=True,
                    title='GO/NO-GO')


# print('Entering debug mode to allow interaction with objects and figures.')
# pdb.set_trace()
