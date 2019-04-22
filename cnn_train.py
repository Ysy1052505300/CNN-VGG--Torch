from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os

import torch

from operator import itemgetter
from cnn_model import CNN
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch import nn


# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './data'


FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    return accuracy

def train(config):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = CIFAR10(DATA_DIR_DEFAULT, train=True, download=True, transform=transform)
    data_loader = DataLoader(train_data, batch_size=config.batch_size)

    model = CNN(3, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # print(batch_inputs.size())
        hit = 0
        n, dim, _, __ = batch_inputs.size()

        # for i in range(n):
        #     temp_x = torch.unsqueeze(batch_inputs[i], 0)
        #     print(temp_x.size())
        #     y_pre = model.forward(temp_x)
        y_pre = model.forward(batch_inputs)
        for i in range(n):
            y_ev, _ = max(enumerate(y_pre[i]), key=itemgetter(1))
            y = batch_targets[i].item()
            if y_ev == y:
                hit += 1

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=10)

        # Add more code here ...
        loss = criterion(y_pre, batch_targets)  # fixme
        accuracy = hit / n * 100  # fixme

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.eval_freq == 0:
            print("loss: ", loss.item())
            print("accuracy: ", accuracy)

        if step == config.max_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


def main(config):
    """
    Main function
    """
    # print(torch.cuda.is_available())
    torch.device("cuda")
    train(config)

if __name__ == '__main__':

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    config = parser.parse_args()
    main(config)