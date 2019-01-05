from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import skimage.transform as im
import tensorflow as tf
from model1 import *
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
parser.add_argument("--epochs", type=int, default=2)
args = parser.parse_args()

validate = False
load_saved_model = False

train_images, train_labels, test_images, test_labels = utils.read_mnist_digits()
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images.shape, test_images.shape)


if validate:
    assert not load_saved_model
    hidden_layers = [[256, 128]]# , [128, 128], [256, 256], [256, 128], [128, 256]]
    for sizes in hidden_layers:
        print("Hidden layer shape =", sizes)
        model = TfLSTM(hidden_layer_sizes=sizes)
        print("Splitting data into training and validation data")
        train_images, validation_images, train_labels, validation_labels =\
            train_test_split(train_images, train_labels, train_size=0.85, random_state=29)
        tik = time.clock()
        model.train(input_images=train_images, input_labels=train_labels,
                    valid_images=validation_images, valid_labels=validation_labels,
                    learning_rate=args.lr, batch_size=args.bs, num_epochs=args.epochs,
                    checkpoint_path=os.path.join(args.model_dir, args.model_name),
                    summary_path=os.path.join(args.log_dir, args.model_name),
                    save_model=False, write_summary=False)
        print("Time taken to complete training = {:.4f} sec\n".format(time.clock() - tik))
else:
    if not load_saved_model:
        model = TfLSTM(hidden_layer_sizes=[256, 128])
        tik = time.clock()
        model.train(input_images=train_images, input_labels=train_labels,
                    # valid_images=test_images, valid_labels=test_labels,
                    learning_rate=args.lr, batch_size=args.bs, num_epochs=args.epochs,
                    checkpoint_path=os.path.join(args.model_dir, args.model_name),
                    summary_path=os.path.join(args.log_dir, args.model_name),
                    save_model=True, write_summary=False)
        acc = model.score(test_images, test_labels)
        print("Average accuracy score in test images = {:.4f}".format(acc))
        print("Time taken to complete training = {:.4f} sec".format(time.clock() - tik))
    else:
        model = TfLSTM(hidden_layer_sizes=[256, 128])
        model.load(model_address)
        print("Model restored")
        acc = model.score(test_images, test_labels)
        print("Average accuracy score in test images = {:.4f}".format(acc))
