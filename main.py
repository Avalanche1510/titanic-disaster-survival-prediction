import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from module.preprocess import *
from module.timer import Timer


def gradient_desc(features, labels, weights, lr, epoch, batch_scale=-1):
    timer = Timer()
    for e in range(epoch):
        epoch_loss = 0
        if batch_scale == -1:
            # full batch
            selected_features = features
            data = TensorDataset(features, labels)
            data = DataLoader(data, batch_size=features.shape[0], shuffle=True)
        else:
            # mini batch
            data = TensorDataset(features, labels)
            data = DataLoader(data, batch_size=batch_scale * 32, shuffle=True)
        for X_batch, y_batch in data:
            # calculate loss
            loss = Loss(X_batch, y_batch, weights)
            epoch_loss += loss
            if loss == np.inf:
                print("gradient explosion takes place!")
                return None
            # trace the computational graph and hence calculate the gradients
            loss.backward()
            # manually update the weights using the gradients calculated
            with torch.no_grad():
                weights -= lr * weights.grad
            # refresh the computational graph and clear the gradients of weights calculated in this epoch
            weights.grad.zero_()
        if e % int(epoch / 16) == 0:
            print(f"epoch: {e + 1}, loss: {epoch_loss/(features.shape[0]//(batch_scale*32)+1)}")
    timer.stop()
    return weights


def Loss(X_batch, y_batch, weights, method="BCE"):
    if method == "BCE":
        # calculate predicted labels
        predict_y = (X_batch @ weights[1:]) + weights[0]
        predict_y = torch.sigmoid(predict_y)
        # set an epsilon so that avoid log0 which is inf
        epsilon = 1e-8
        # calculate BCE loss using formula BCELoss(y,#y) = -[y * log(#y) + (1-y) * log(1-#y)] where #y is the predicted labels
        loss = -(y_batch.t()*torch.log(predict_y+epsilon) + ((1-y_batch).t())*torch.log(1-predict_y+epsilon))
        loss = loss.mean()
        return loss
    else:
        return None


def main(gpu=False):
    np.random.seed(42)
    # switch device: gpu/cpu
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"device using: {device}")

    # initialise data
    train_features, train_labels = to_tensor("train")
    test_features, test_labels = to_tensor("test")
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    # test_features = train_features.to(device)
    # test_labels = train_features.to(device)

    # there's 10 weights for 10 features and 1 bias, in total 11 weights.
    weights = torch.tensor(np.random.randn(11, 1), dtype=torch.float32).to(device)
    weights.requires_grad_()
    print(gradient_desc(train_features, train_labels, weights, 0.001, 5000, 1))


if __name__ == "__main__":
    main(False)
