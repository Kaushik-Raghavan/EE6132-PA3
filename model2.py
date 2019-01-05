from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
import numpy as np
import random
import joblib
import struct
import utils
import time
import os


FREQUENCY_OF_SUMMARY_UPDATE = 10
FREQUENCY_OF_VERBOSE_UPDATE = 100


class Model:

    def __init__(self, activation_fn=tf.nn.relu, input_dims=(28, 28), output_dim=10):
        self.input = None
        self.activation_fn = activation_fn
        self.comp_graph = tf.Graph()
        with self.comp_graph.as_default():
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_in_training_mode')
            self.image_batch = tf.placeholder(dtype=tf.float32, shape=(None,) + input_dims, name='feed_input')
            self.label_batch = tf.placeholder(dtype=tf.float32, shape=(None,), name='feed_labels')
            self.sequence_len = tf.placeholder(dtype=tf.int32, shape=(None,), name='sequence_lengths')

    def build(self, hidden_layer_sizes, output_dim=10):
        pass

    def get_loss(self, predictions, gt_labels):
        with self.comp_graph.as_default():
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(gt_labels, dtype=tf.int32),
                                                              logits=predictions, name='cross_entropy_loss')
            loss_mean = tf.reduce_mean(loss, name='mean_cross_entropy_loss')
            return loss_mean

    def self_train(self, generating_fn, generate_on_the_fly=True, idx=3, learning_rate=1e-4, batch_size=64, num_iter=1000,
                   valid_images=None, valid_labels=None, valid_seq_length=None,
                   summary_path='logs/', checkpoint_path="./models/", save_model=False, write_summary=False):
        with self.comp_graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam_optimizer')
                train_op = opt.minimize(self.loss_op, global_step=tf.train.get_global_step(), name='training_operations')
            saver = tf.train.Saver(max_to_keep=5)

        with self.comp_graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())

        ckpt_step = num_iter // FREQUENCY_OF_SUMMARY_UPDATE
        if num_iter % FREQUENCY_OF_SUMMARY_UPDATE is not 0:
            ckpt_step += 1
        verbose_step = num_iter // FREQUENCY_OF_VERBOSE_UPDATE
        if num_iter % FREQUENCY_OF_VERBOSE_UPDATE is not 0:
            verbose_step += 1

        training_error, test_error = [], []
        if not generate_on_the_fly:
            training_data = generating_fn(64000, k=idx)
            training_data = list(zip(training_data[0], training_data[1], training_data[2]))

        for step in range(num_iter):
            if not generate_on_the_fly:
                data = [None, None, None]
                if step % 1000 == 0:
                    random.shuffle(training_data)
                    shuffled_data = list(zip(*training_data))
                    shuffled_data[0] = np.array(shuffled_data[0])
                    shuffled_data[1] = np.array(shuffled_data[1])
                    shuffled_data[2] = np.array(shuffled_data[2])
                    print(shuffled_data[0].shape, shuffled_data[1].shape, shuffled_data[2].shape)
                st_idx = (step * batch_size) % 64000
                end_idx = st_idx + batch_size
                data[0] = shuffled_data[0][st_idx: end_idx]
                data[1] = shuffled_data[1][st_idx: end_idx]
                data[2] = shuffled_data[2][st_idx: end_idx]
            else:
                data = generating_fn(batch_size, k=idx)

            if (step + 1) % verbose_step is 0:
                feed_dict = {self.image_batch: data[0], self.label_batch: data[1], self.sequence_len: data[2]}
                _, loss = self.sess.run([train_op, self.loss_op], feed_dict=feed_dict)
                verbose = "Step = {}: Training loss = {:.5f}".format(step + 1, loss)
                if valid_images is not None and valid_labels is not None:
                    feed_dict = {self.image_batch: valid_images, self.label_batch: valid_labels, self.sequence_len: valid_seq_length}
                    valid_accuracy, valid_loss = self.sess.run([self.accuracy_op, self.loss_op], feed_dict=feed_dict)
                    valid_verbose = "Validation loss = {:.5f}; Validation accuracy = {:.4f}".format(valid_loss, valid_accuracy)
                    verbose = verbose + '; ' + valid_verbose
                    test_error.append([step, valid_accuracy])
                print(verbose)
            else:
                feed_dict = {self.image_batch: data[0], self.label_batch: data[1], self.sequence_len: data[2]}
                _, loss = self.sess.run([train_op, self.loss_op], feed_dict=feed_dict)

            training_error.append([step, loss])

            if (step + 1) % ckpt_step is 0:
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                if save_model:
                    with self.comp_graph.as_default():
                        saver.save(self.sess, os.path.join(checkpoint_path, 'model'), global_step=step)
                    print("Checkpoint created after {} iterations".format(step + 1))

        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        # np.savetxt(os.path.join(summary_path, "LSTM_training_error.txt"), training_error)
        # np.savetxt(os.path.join(summary_path, "LSTM_test_accuracy.txt"), test_error)

    def score(self, test_images, test_labels, seq_length=None):
        with self.comp_graph.as_default():
            feed_dict = {self.image_batch: test_images, self.label_batch: test_labels,
                         self.sequence_len: seq_length, self.is_training: False}
            test_accuracy = self.sess.run(self.accuracy_op, feed_dict=feed_dict)
        return test_accuracy

    def predict(self, images, one_hot_output=False, seq_length=None):
        with self.comp_graph.as_default():
            if one_hot_output:
                logits = tf.nn.softmax(self.output)
                feed_dict = {self.image_batch: images, self.sequence_len: seq_length, self.is_training: False}
                out = self.sess.run(logits, feed_dict=feed_dict)
                return out
            feed_dict = {self.image_batch: images, self.sequence_len: seq_length, self.is_training: False}
            out = self.sess.run(self.output, feed_dict=feed_dict)
            print(out.shape)
            return np.argmax(out, axis=1)

    def load(self, model_dir):
        """ Loads the model stored at the latest check point in the given directory """
        with self.comp_graph.as_default():
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            saver = tf.train.Saver()
            self.sess.run(tf.local_variables_initializer())
            saver.restore(self.sess, latest_checkpoint)

    def get_filter_val(self, layer_name):
        kernel_val, bias_val = None, None
        kernel_val = self.comp_graph.get_tensor_by_name(layer_name + '/kernel:0').eval(session=self.sess)
        bias_val = self.comp_graph.get_tensor_by_name(layer_name + '/bias:0').eval(session=self.sess)
        return kernel_val, bias_val

    def get_bn_params(self, layer_name):
        gamma, beta = None, None
        gamma = self.comp_graph.get_tensor_by_name(layer_name + '/gamma:0').eval(session=self.sess)
        beta = self.comp_graph.get_tensor_by_name(layer_name + '/beta:0').eval(session=self.sess)
        moving_mean = self.comp_graph.get_tensor_by_name(layer_name + '/moving_mean:0').eval(session=self.sess)
        moving_var = self.comp_graph.get_tensor_by_name(layer_name + '/moving_variance:0').eval(session=self.sess)
        return gamma, beta, moving_mean, moving_var

    def plot_filters(self, layer_name, filter_idx=0, plot_title=None):
        conv_kernel_val, conv_bias_val = self.get_filter_val(layer_name)
        conv_kernel_val = np.squeeze(conv_kernel_val[:, :, filter_idx, :])

        if plot_title is None:
            plot_title = layer_name

        vals = np.transpose(conv_kernel_val, axes=(2, 0, 1))
        fig, axes = plt.subplots(4, 8, figsize=(12, 6), subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.2, wspace=0.0))
        fig.suptitle(plot_title, fontsize=16)
        for i, ax in enumerate(axes.flat):
            new_img = im.resize(vals[i].reshape(3,3), (10, 10))
            ax.imshow(new_img, cmap='hot')
        plt.show()


class TfRNN(Model):
    def __init__(self, hidden_layer_sizes, activation_fn=tf.nn.relu, input_dims=(28,28), output_dim=10):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dim=output_dim)
        self.build(hidden_layer_sizes=hidden_layer_sizes, output_dim=output_dim)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, hidden_layer_sizes, output_dim=10):
        with self.comp_graph.as_default():
            cells = [tf.nn.rnn_cell.BasicRNNCell(num_units=n) for n in hidden_layer_sizes]
            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            initial_state = stacked_cells.zero_state(tf.shape(self.image_batch)[0], dtype=tf.float32)
            with tf.variable_scope("RNN_units"):
                self.output, state = tf.nn.dynamic_rnn(stacked_cells, self.image_batch, initial_state=initial_state, sequence_length=self.sequence_len)
            dim1_idx = self.sequence_len - tf.ones(shape=tf.shape(self.sequence_len), dtype=tf.int32)
            dim0_idx = tf.range(tf.shape(self.sequence_len)[0], dtype=tf.int32)
            output_idx = tf.transpose(tf.stack([dim0_idx, dim1_idx]))
            self.output = tf.layers.dense(inputs=tf.gather_nd(self.output, output_idx), units=output_dim, activation=None, name='V')

            self.loss_op = self.get_loss(self.output, self.label_batch)
            pred_classes = tf.argmax(self.output, axis=1)
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.label_batch, predictions=pred_classes, name='accuracy_op')


class TfLSTM(Model):
    def __init__(self, hidden_layer_sizes, activation_fn=tf.nn.relu, input_dims=(28,28), output_dim=10):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dim=output_dim)
        self.build(hidden_layer_sizes=hidden_layer_sizes, output_dim=output_dim)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, hidden_layer_sizes, output_dim=10):
        with self.comp_graph.as_default():
            cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in hidden_layer_sizes]
            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            initial_state = stacked_cells.zero_state(tf.shape(self.image_batch)[0], dtype=tf.float32)
            with tf.variable_scope("LSTM_units"):
                self.output, state = tf.nn.dynamic_rnn(stacked_cells, self.image_batch, initial_state=initial_state, sequence_length=self.sequence_len)
            dim1_idx = self.sequence_len - tf.ones(shape=tf.shape(self.sequence_len), dtype=tf.int32)
            dim0_idx = tf.range(tf.shape(self.sequence_len)[0], dtype=tf.int32)
            output_idx = tf.transpose(tf.stack([dim0_idx, dim1_idx]))
            self.output = tf.layers.dense(inputs=tf.gather_nd(self.output, output_idx), units=output_dim, activation=None, name='V')
            print(self.output.shape)

            self.loss_op = self.get_loss(self.output, self.label_batch)
            pred_classes = tf.argmax(self.output, axis=1)
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.label_batch, predictions=pred_classes, name='accuracy_op')


class TfBidirectionalRNN(Model):
    def __init__(self, hidden_layer_sizes, activation_fn=tf.nn.relu, input_dims=(28,28), output_dim=10):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dim=output_dim)
        self.build(hidden_layer_sizes=hidden_layer_sizes, output_dim=output_dim)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, hidden_layer_sizes, output_dim=10):
        with self.comp_graph.as_default():
            inputs_series = tf.unstack(self.image_batch, axis=1)
            cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, name='forward_pass') for n in hidden_layer_sizes]
            stacked_cells_fwd = tf.nn.rnn_cell.MultiRNNCell(cells)
            cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n, name='backward_pass') for n in hidden_layer_sizes]
            stacked_cells_rev = tf.nn.rnn_cell.MultiRNNCell(cells)
            with tf.variable_scope("BidirectionalRNN_units"):
                states1 = []
                state = stacked_cells_fwd.zero_state(tf.shape(self.image_batch)[0], dtype=tf.float32)
                for current_input in inputs_series:
                    output1, state = stacked_cells_fwd(current_input, state)
                    states1.append(output1)
                states2 = []
                state = stacked_cells_rev.zero_state(tf.shape(self.image_batch)[0], dtype=tf.float32)
                for current_input in reversed(inputs_series):
                    output2, state = stacked_cells_rev(current_input, state)
                    states2.append(output2)

                self.output = tf.layers.dense(inputs=tf.concat([states1[-1], states2[-1]], axis=1), units=output_dim, activation=None, name='V')

            self.loss_op = self.get_loss(self.output, self.label_batch)
            pred_classes = tf.argmax(self.output, axis=1)
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.label_batch, predictions=pred_classes, name='accuracy_op')


class VanillaRNN(Model):
    def __init__(self, hidden_layer_sizes, activation_fn=tf.nn.relu, input_dims=(28,28), output_dim=10):
        Model.__init__(self, activation_fn=activation_fn, input_dims=input_dims, output_dim=output_dim)
        self.build(hidden_layer_sizes=hidden_layer_sizes, output_dim=output_dim)
        self.sess = tf.Session(graph=self.comp_graph)

    def build(self, hidden_layer_sizes, output_dim=10):
        with self.comp_graph.as_default():
            inputs_series = tf.unstack(self.image_batch, axis=1)
            with tf.variable_scope("RNN_units"):
                hidden_state = tf.zeros(shape=(tf.shape(self.image_batch)[0], hidden_layer_size), name='initial_hidden_state')
                input_and_state = tf.concat([inputs_series[0], hidden_state], axis=1)
                hidden_state = tf.layers.dense(inputs=input_and_state, units=128, activation=tf.nn.tanh, name='U')
                self.output = tf.layers.dense(inputs=hidden_state, units=output_dim, activation=None, name='V')
                for current_input in inputs_series[1:]:
                    input_and_state = tf.concat([current_input, hidden_state], axis=1)
                    hidden_state = tf.layers.dense(inputs=input_and_state, units=128, activation=self.activation_fn, name='U', reuse=True)
                    self.output = tf.layers.dense(inputs=hidden_state, units=output_dim, activation=None, name='V', reuse=True)

            self.loss_op = self.get_loss(self.output, self.label_batch)
            pred_classes = tf.argmax(self.output, axis=1)
            _, self.accuracy_op = tf.metrics.accuracy(labels=self.label_batch, predictions=pred_classes, name='accuracy_op')
