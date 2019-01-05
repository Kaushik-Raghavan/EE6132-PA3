from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
from model3 import *
import numpy as np
import argparse
import random
import joblib
import struct
import utils
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--load_model", action='store_true')
parser.add_argument("--write_summary", action='store_true')
parser.add_argument("--save_model", action='store_true')
parser.add_argument("--model_name", type=str, default='unknown')
parser.add_argument("--model_dir", type=str, default="./models")
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--iter", type=int, default=1000)
args = parser.parse_args()

np.random.seed(int(time.clock() * 10000))


def generating_fn(batch_size, L=5):
    once = True
    data = []
    labels = []
    lengths = []
    max_num = int(2**L)
    for _ in range(batch_size):
        num = np.random.randint(low=0, high=max_num, size=2)
        inp_bits = [np.binary_repr(num[i], width=L+1) for i in range(2)]
        outp_bits = np.binary_repr(num[0] + num[1], width=L+1)
        inp = [[int(x), int(y)] for x, y in zip(inp_bits[0][::-1], inp_bits[1][::-1])]
        outp = [int(x) for x in outp_bits[::-1]]
        data.append(inp)
        labels.append(outp)
        lengths.append(L + 1)
    return (np.array(data), np.array(labels), np.array(lengths))


if not args.load_model:
    seq_L = [5]
    for L in seq_L:
        print("L = {}".format(L))
        model = TfLSTM(hidden_layer_sizes=[5], input_dims=(None, 2), label_dims=(None,), output_units=1,
                       output_activ=lambda x: tf.where(tf.greater_equal(tf.nn.sigmoid(x), 0.5), tf.ones(shape=tf.shape(x)), tf.zeros(shape=tf.shape(x))))
        tik = time.clock()
        test_images, test_labels, seq_length = generating_fn(10000, L=L)
        print(test_images.shape, test_labels.shape)
        model.self_train(generating_fn=generating_fn, L=L,
                         valid_images=test_images, valid_labels=test_labels, valid_seq_length=seq_length,
                         learning_rate=args.lr, batch_size=args.bs, num_iter=args.iter,
                         checkpoint_path=os.path.join(args.model_dir, args.model_name),
                         summary_path=os.path.join(args.log_dir, args.model_name),
                         save_model=True, write_summary=False)
        acc = model.score(test_images, test_labels, seq_length=seq_length)
        print("Average accuracy score in test images = {:.4f}".format(acc))
        print("Time taken to complete training = {:.4f} sec".format(time.clock() - tik))
else:
    model = TfLSTM(hidden_layer_sizes=[5], input_dims=(None, 2), label_dims=(None,), output_units=1,
                   output_activ=lambda x: tf.where(tf.greater_equal(tf.nn.sigmoid(x), 0.5), tf.ones(shape=tf.shape(x)), tf.zeros(shape=tf.shape(x))))
    model_address = os.path.join(args.model_dir, args.model_name)
    model.load(model_address)
    print("Model restored")
    np.random.seed(int(time.clock() * 10000))
    seq_L = np.arange(1, 21).astype(np.int32)
    accuracies = []
    for L in seq_L:
        test_images, test_labels, seq_length = generating_fn(1000, L=L)
        acc = model.score(test_images, test_labels, seq_length=seq_length)
        accuracies.append(acc)

    print(accuracies)
    data = [[x, y] for x, y in zip(seq_L, np.array(accuracies))]
    np.savetxt('different_L.txt', data)
