'''
flipflop.py
Written using Python 2.7.12
@ Matt Golub, August 2018.
Please direct correspondence to mgolub@stanford.edu.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
from RecurrentWhisperer import RecurrentWhisperer


class DualTask(RecurrentWhisperer):
    ''' Class for training an RNN to perform the dual task described in
        Zhang et al. 2018 bioRxiv

    Task:
        Briefly, the task is composed of two nested tasks: a delay pair
        association task that requires the RNN to compare two stimuli
        presented with a delay, and a simple Go-NoGo task that works as
        a distractor.

        This class generates synthetic data for the task via
        generate_dualtask_trials(...).

    Usage:
        This class trains an RNN to generate the correct outputs given the
        inputs of the dual-task. All that is needed to get started is to
        construct a dualtask object and to call .train on that object:

        # dict of hyperparameter key/value pairs
        # (see 'Hyperparameters' section below)
        hps = {...}

        ff = DualTask(**hps)
        ff.train()

    Hyperparameters:
        rnn_type: string specifying the architecture of the RNN. Currently
        must be one of {'vanilla', 'gru', 'lstm'}. Default: 'vanilla'.

        n_hidden: int specifying the number of hidden units in the RNN.
        Default: 24.

        data_hps: dict containing hyperparameters for generating synthetic
        data. Contains the following keys:

            'n_batch': int specifying the number of synthetic trials to use
            per training batch (i.e., for one gradient step). Default: 1024.

            'n_time': int specifying the duration of each synthetic trial
            (measured in timesteps). Default: 32.

            'n_bits': int specifying the number of input channels into the
            FlipFlop device (which will also be the number of output channels).
            Default: 6 (corresponding to the six stimuli).

            'gng_time': time at which the go-noGo stimulus will be presented.
            Should be smaller than n_time

            'noise': std of gaussian noise added independently to each channel.


            'lamb': parametrization of the stimulus S5, S6. Default: 0

            'delay_max': Maximum delay of appearence of the dpa2. Should be
            between  Default: 5

        log_dir: string specifying the top-level directory for saving various
        training runs (where each training run is specified by a different set
        of hyperparameter settings). When tuning hyperparameters, log_dir is
        meant to be constant across models. Default: '/tmp/dualtask_logs/'.

        n_trials_plot: int specifying the number of synthetic trials to plot
        per visualization update. Default: 4.
    '''

    @staticmethod
    def _default_hash_hyperparameters():
        '''Defines default hyperparameters, specific to DualTask, for the set
        of hyperparameters that are hashed to define a directory structure for
        easily managing multiple runs of the RNN training (i.e., using
        different hyperparameter settings). Additional default hyperparameters
        are defined in RecurrentWhisperer (from which DualTask inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'rnn_type': 'vanilla',
            'n_hidden': 24,
            'data_hps': {
                'n_batch': 1028,
                'n_time': 32,
                'n_bits': 6,
                'gng_time': 10,
                'noise': 0.1,
                'lamb': 0.25,
                'delay_max': 5}
            }

    @staticmethod
    def _default_non_hash_hyperparameters():
        '''Defines default hyperparameters, specific to DualTask, for the set
        of hyperparameters that are NOT hashed. Additional default
        hyperparameters are defined in RecurrentWhisperer (from which DualTask
        inherits).

        Args:
            None.

        Returns:
            dict of hyperparameters.
        '''
        return {
            'log_dir': '/tmp/dualtask_logs/',
            'n_trials_plot': 6,
            }

    def _setup_model(self):
        '''Defines an RNN in Tensorflow.

        See docstring in RecurrentWhisperer.
        '''
        hps = self.hps
        n_hidden = hps.n_hidden

        data_hps = hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_inputs = data_hps['n_bits']
        n_output = n_inputs

        # Data handling
        self.inputs_bxtxd = tf.placeholder(tf.float32,
                                           [n_batch, n_time, n_inputs])
        self.output_bxtxd = tf.placeholder(tf.float32,
                                           [n_batch, n_time, n_output])

        # RNN
        if hps.rnn_type == 'vanilla':
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        elif hps.rnn_type == 'gru':
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
        elif hps.rnn_type == 'lstm':
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        else:
            raise ValueError('Hyperparameter rnn_type must be one of '
                             '[vanilla, gru, lstm] but was %s' % hps.rnn_type)

        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)
        self.hidden_bxtxd, _ = tf.nn.dynamic_rnn(self.rnn_cell,
                                                 self.inputs_bxtxd,
                                                 initial_state=initial_state)

        # Readout from RNN
        np_W_out, np_b_out = self._np_init_weight_matrix(n_hidden, n_output)
        self.W_out = tf.Variable(np_W_out, dtype=tf.float32)
        self.b_out = tf.Variable(np_b_out, dtype=tf.float32)
        self.pred_output_bxtxd = tf.tensordot(self.hidden_bxtxd,
                                              self.W_out, axes=1) + self.b_out

        # Loss
        # self.loss =\
        # tf.nn.sigmoid_cross_entropy_with_logits(labels=self.output_bxtxd,
        # logits=self.pred_output_bxtxd)

        self.loss = tf.reduce_mean(
                tf.squared_difference(self.output_bxtxd,
                                      self.pred_output_bxtxd))

    def _setup_saver(self):
        '''See docstring in RecurrentWhisperer.'''

        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.hps.max_ckpt_to_keep)

    def _setup_training(self, train_data, valid_data):
        '''Does nothing. Required by RecurrentWhisperer.'''
        pass

    def _train_batch(self, batch_data):
        '''Performs a training step over a single batch of data.

        Args:
            batch_data: dict containing one training batch of data. Contains
            the following key/value pairs:

                'inputs': [n_batch x n_time x n_bits] numpy array specifying
                the inputs to the RNN.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct output responses to the 'inputs.'

        Returns:
            summary: dict containing the following summary key/value pairs
            from the training step:

                'loss': scalar float evalutaion of the loss function over the
                data batch.

                'grad_global_norm': scalar float evaluation of the norm of
                the gradient of the loss function with respect to all trainable
                variables, taken over the data batch.
        '''

        ops_to_eval = [self.train_op,
                       self.grad_global_norm,
                       self.loss,
                       self.merged_opt_summary]

        feed_dict = dict()
        feed_dict[self.inputs_bxtxd] = batch_data['inputs']
        feed_dict[self.output_bxtxd] = batch_data['output']
        feed_dict[self.learning_rate] = self.adaptive_learning_rate()
        feed_dict[self.grad_norm_clip_val] = self.adaptive_grad_norm_clip()

        [ev_train_op,
         ev_grad_global_norm,
         ev_loss,
         ev_merged_opt_summary] = self.session.run(ops_to_eval,
                                                   feed_dict=feed_dict)

        if self.hps.do_save_tensorboard_events:

            if self._epoch() == 0:
                '''Hack to prevent throwing the vertical axis on the
                Tensorboard figure for grad_norm_clip_val (grad_norm_clip val
                is initialized to an enormous number to prevent clipping
                before we know the scale of the gradients).'''
                feed_dict[self.grad_norm_clip_val] = np.nan
                ev_merged_opt_summary = \
                    self.session.run(self.merged_opt_summary, feed_dict)

            self.writer.add_summary(ev_merged_opt_summary, self._step())

        summary = {'loss': ev_loss, 'grad_global_norm': ev_grad_global_norm}

        return summary

    def predict(self, batch_data, do_predict_full_LSTM_state=False):
        '''Runs the RNN given its inputs.

        Args:
            batch_data:
                dict containing the key 'inputs': [n_batch x n_time x n_bits]
                numpy array specifying the inputs to the RNN.

            do_predict_full_LSTM_state (optional): bool indicating, if the RNN
            is an LSTM, whether to return the concatenated hidden and cell
            states (True) or simply the hidden states (False). Default: False.

        Returns:
            predictions: dict containing the following key/value pairs:

                'state': [n_batch x n_time x n_states] numpy array containing
                the activations of the RNN units in response to the inputs.
                Here, n_states is the dimensionality of the hidden state,
                which, depending on the RNN architecture and
                do_predict_full_LSTM_state, may or may not include LSTM cell
                states.

                'output': [n_batch x n_time x n_bits] numpy array containing
                the readouts from the RNN.

        '''

        if do_predict_full_LSTM_state:
            return self._predict_with_LSTM_cell_states(batch_data)
        else:
            ops_to_eval = [self.hidden_bxtxd, self.pred_output_bxtxd]
            feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
            ev_hidden_bxtxd, ev_pred_output_bxtxd = \
                self.session.run(ops_to_eval, feed_dict=feed_dict)

            predictions = {
                'state': ev_hidden_bxtxd,
                'output': ev_pred_output_bxtxd
                }

            return predictions

    def _predict_with_LSTM_cell_states(self, batch_data):
        '''Runs the RNN given its inputs.

        The following is added for execution only when LSTM predictions are
        needed for both the hidden and cell states. Tensorflow does not make
        it easy to access the cell states via dynamic_rnn.

        Args:
            batch_data: as specified by predict.

        Returns:
            predictions: as specified by predict.

        '''

        hps = self.hps
        if hps.rnn_type != 'lstm':
            return self.predict(batch_data)

        n_hidden = hps.n_hidden
        [n_batch, n_time, n_bits] = batch_data['inputs'].shape
        initial_state = self.rnn_cell.zero_state(n_batch, dtype=tf.float32)

        ''' Add ops to the graph for getting the complete LSTM state
        (i.e., hidden and cell) at every timestep.'''
        self.full_state_list = []
        for t in range(n_time):
            input_ = self.inputs_bxtxd[:, t, :]
            if t == 0:
                full_state_t_minus_1 = initial_state
            else:
                full_state_t_minus_1 = self.full_state_list[-1]
            _, full_state_bxd = self.rnn_cell(input_, full_state_t_minus_1)
            self.full_state_list.append(full_state_bxd)

        '''Evaluate those ops'''
        ops_to_eval = [self.full_state_list, self.pred_output_bxtxd]
        feed_dict = {self.inputs_bxtxd: batch_data['inputs']}
        ev_full_state_list, ev_pred_output_bxtxd = \
            self.session.run(ops_to_eval, feed_dict=feed_dict)

        '''Package the results'''
        h = np.zeros([n_batch, n_time, n_hidden])  # hidden states: bxtxd
        c = np.zeros([n_batch, n_time, n_hidden])  # cell states: bxtxd
        for t in range(n_time):
            h[:, t, :] = ev_full_state_list[t].h
            c[:, t, :] = ev_full_state_list[t].c

        ev_LSTMCellState = tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c)

        predictions = {
            'state': ev_LSTMCellState,
            'output': ev_pred_output_bxtxd
            }

        return predictions

    def _get_data_batches(self, train_data):
        '''See docstring in RecurrentWhisperer.'''
        return [self.generate_dualtask_trials()]

    def _get_batch_size(self, batch_data):
        '''See docstring in RecurrentWhisperer.'''
        return batch_data['inputs'].shape[0]

    def generate_dualtask_trials(self):
        '''Generates synthetic data (i.e., ground truth trials) for the
        dual task. See comments following DualTask class definition for a
        description of the input-output relationship in the task.

        Args:
            None.

        Returns:
            dict containing 'inputs' and 'outputs'.

                'inputs': [n_batch x n_time x n_bits] numpy array containing
                input pulses.

                'outputs': [n_batch x n_time x n_bits] numpy array specifying
                the correct behavior of the Dual task device.
        '''

        data_hps = self.hps.data_hps
        n_batch = data_hps['n_batch']
        n_time = data_hps['n_time']
        n_bits = data_hps['n_bits']
        gng_time = data_hps['gng_time']
        lamb = data_hps['lamb']
        delay_max = data_hps['delay_max']

        dataset = data.get_inputs_outputs(n_batch, n_time,
                                          n_bits, gng_time, lamb, delay_max)
        return dataset

    def _setup_visualizations(self):
        '''See docstring in RecurrentWhisperer.'''
        FIG_WIDTH = 6  # inches
        FIX_HEIGHT = 9  # inches
        self.fig = plt.figure(figsize=(FIG_WIDTH, FIX_HEIGHT),
                              tight_layout=True)

    def _update_visualizations(self, train_data=None, valid_data=None):
        '''See docstring in RecurrentWhisperer.'''
        data = self.generate_dualtask_trials()
        self.plot_trials(data)

    def plot_trials(self, data, start_time=0, stop_time=None):
        '''Plots example trials, complete with input pulses, correct outputs,
        and RNN-predicted outputs.

        Args:
            data: dict as returned by generate_dualtask_trials.

            start_time (optional): int specifying the first timestep to plot.
            Default: 0.

            stop_time (optional): int specifying the last timestep to plot.
            Default: n_time.

        Returns:
            None.
        '''
        hps = self.hps
        n_batch = self.hps.data_hps['n_batch']
        n_time = self.hps.data_hps['n_time']
        n_plot = np.min([hps.n_trials_plot, n_batch])
        dpa2_time = data['vec_tau']

        f = plt.figure(self.fig.number)
        plt.clf()

        inputs = data['inputs']
        output = data['output']
        predictions = self.predict(data)
        pred_output = predictions['output']

        if stop_time is None:
            stop_time = n_time

        time_idx = range(start_time, stop_time)

        for trial_idx in range(n_plot):
            plt.subplot(n_plot, 1, trial_idx+1)
            if n_plot == 1:
                plt.title('Example trial', fontweight='bold')
            else:
                plt.title('Example trial %d | %d' % (trial_idx + 1, dpa2_time[trial_idx]),
                          fontweight='bold')

            self._plot_single_trial(
                inputs[trial_idx, time_idx, :],
                output[trial_idx, time_idx, :],
                pred_output[trial_idx, time_idx, :])

            # Only plot x-axis ticks and labels on the bottom subplot
            if trial_idx < (n_plot-1):
                plt.xticks([])
            else:
                plt.xlabel('Timestep', fontweight='bold')

        plt.ion()
        plt.show()
        plt.pause(1e-10)
        return f

    @staticmethod
    def _plot_single_trial(input_txd, output_txd, pred_output_txd):

        VERTICAL_SPACING = 2.5
        [n_time, n_bits] = input_txd.shape
        tt = range(n_time)

        y_ticks = [VERTICAL_SPACING*bit_idx for bit_idx in range(n_bits)]
        y_tick_labels = ['S %d' % (bit_idx+1) for bit_idx in range(n_bits)]
        plt.yticks(y_ticks, y_tick_labels, fontweight='bold')
        for bit_idx in range(n_bits):

            vertical_offset = VERTICAL_SPACING*bit_idx

            # Input pulses
            plt.fill_between(
                tt,
                vertical_offset + input_txd[:, bit_idx],
                vertical_offset,
                step='mid',
                color='gray')

            # Correct outputs
            plt.step(
                tt,
                vertical_offset + output_txd[:, bit_idx],
                where='mid',
                linewidth=2,
                color='cyan')

            # RNN outputs
            plt.step(
                tt,
                vertical_offset + pred_output_txd[:, bit_idx],
                where='mid',
                color='purple',
                linewidth=1.5,
                linestyle='--')

        plt.xlim(-1, n_time)
