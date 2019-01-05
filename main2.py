from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
from model2 import *
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

def latex_print(outp, labels):
    strout = "\makebox{Sequence} & "
    cnt = 0
    for x, y, l in outp:
        cnt += 1
        seq = np.array(x)
        seq = np.argmax(seq, axis=1)[:l]
        strout += "["
        for i in range(l):
            if i == idx_k - 1: strout += "\\textbf{%d}" % seq[i]
            else: strout += "%d" % seq[i]
            if i != l - 1: strout += ","
        strout += "]"
        if cnt != len(outp): strout += " & "
    strout += "\\\\\\hline"
    print(strout)
    strout = "\makebox{Predicted output} & "
    cnt = 0
    for _, y, _ in outp:
        cnt += 1
        seq = np.array(x)
        seq = np.argmax(seq, axis=1)[:l]
        strout += "%d" % y
        if cnt != len(outp): strout += " & "
    strout += "\\\\\\hline"
    print(strout)



def generating_fn(size, rangeL=10, k=3):
    assert rangeL >= k, "k should be less than the upper limit of length of sequence - specified by rangeL"
    data = []
    labels = []
    lengths = []
    for _ in range(size):
        curr_size = np.random.randint(low=k, high=rangeL, size=None)
        arr = np.random.randint(low=0, high=10, size=rangeL)
        one_hot = utils.one_hot(arr, size=10)
        data.append(one_hot)
        labels.append(arr[k - 1])
        lengths.append(curr_size)
    return (np.array(data), np.array(labels), np.array(lengths))


if not args.load_model:
    seq_k = [7]
    for k in seq_k:
        print("k = {}".format(k))
        model = TfLSTM(hidden_layer_sizes=[10], input_dims=(None, 10))
        tik = time.clock()
        test_images, test_labels, seq_length = generating_fn(size=10000, k=k)
        model.self_train(generating_fn=generating_fn, idx=k,
                        valid_images=test_images, valid_labels=test_labels, valid_seq_length=seq_length,
                        learning_rate=args.lr, batch_size=args.bs, num_iter=args.iter,
                        checkpoint_path=os.path.join(args.model_dir, args.model_name + "_{}".format(k)),
                        summary_path=os.path.join(args.log_dir, args.model_name),
                        save_model=True, write_summary=False)
        acc = model.score(test_images, test_labels, seq_length=seq_length)
        print("Average accuracy score in test images = {:.4f}".format(acc))
        print("Time taken to complete training = {:.4f} sec".format(time.clock() - tik))
else:
    model = TfLSTM(hidden_layer_sizes=[10], input_dims=(None, 10))
    model_address = os.path.join(args.model_dir, args.model_name)
    model.load(model_address)
    print("Model restored")
    idx_k = 7
    test_images, test_labels, seq_length = generating_fn(size=10000, k=idx_k)

    pred = model.predict(test_images, seq_length=seq_length)
    wrong_pred_idx = np.where(pred != test_labels)[0]
    correct_pred_idx = np.where(pred == test_labels)[0]
    print("# incorrect predictions =", len(wrong_pred_idx))
    outp = []
    ## Chosing random samples from test dataset
    [outp.append([x, y, l]) for x, y, l in zip(test_images[wrong_pred_idx[18:21]], pred[wrong_pred_idx[18:21]], seq_length[wrong_pred_idx[18:21]])]
    [outp.append([x, y, l]) for x, y, l in zip(test_images[correct_pred_idx[18:23]], pred[correct_pred_idx[18:23]], seq_length[correct_pred_idx[18:23]])]
    random.shuffle(outp)
    # latex_print(outp)

    acc = model.score(test_images, test_labels, seq_length=seq_length)
    print("Average accuracy score in test images = {:.4f}".format(acc))
