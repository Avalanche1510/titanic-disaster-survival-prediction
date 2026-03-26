import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from module.preprocess import *
from module.timer import Timer
from module.normalization import *
from module.evaluation import *


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
            # calculate weighted loss for this epoch as process
            epoch_loss += loss * (y_batch.shape[0] / labels.shape[0])
            # trace the computational graph and hence calculate the gradients
            loss.backward()
            # manually update the weights using the gradients calculated
            with torch.no_grad():
                weights -= lr * weights.grad
            # refresh the computational graph and clear the gradients of weights calculated in this epoch
            grad = weights.grad.clone()
            weights.grad.zero_()
        if e % int(epoch / 16) == 0:
            print(f"epoch: {e + 1}, loss: {epoch_loss}")
            print(evaluate(weights, features, labels))
    timer.stop()
    return weights


def Loss(X_batch, y_batch, weights, method="BCE_logits_manual"):
    if method == "BCE":
        # calculate predicted labels
        predict_y = X_batch @ weights[1:] + weights[0]
        predict_y = torch.sigmoid(predict_y)
        # set an epsilon so that avoid log0 which is inf
        epsilon = 1e-8
        # calculate BCE loss using formula BCELoss(y,#y) = -[y * log(#y) + (1-y) * log(1-#y)] where #y is the predicted labels
        loss = -(y_batch.T*torch.log(predict_y+epsilon) + (1-y_batch).T*torch.log(1-predict_y+epsilon))
        loss = loss.mean()
        return loss
    elif method == "BCE_logits":
        logits = X_batch @ weights[1:] + weights[0]  # 不要先 sigmoid
        print("logits: ", logits)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(logits, y_batch)
        return loss
    elif method == "BCE_logits_manual":
        # manual version of BCE with Logits, avoid log0
        logits = X_batch @ weights[1:] + weights[0]
        # formula: loss = max(0,x)− x*y + log(1+e^-|x|)
        loss = torch.where(logits >= 0, logits, torch.zeros_like(logits)) - logits * y_batch + torch.log(1 + torch.exp(-(logits.abs())))
        # sum up the tensor into a scalar
        loss = loss.mean()
        return loss
    else:
        return None


def main(gpu=False):
    np.random.seed(24)
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
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    # normalize using z-score
    validate_features = train_features.clone()
    train_features, mean, std = normalise(train_features, "z-score")

    # there's 10 weights for 10 features and 1 bias, in total 11 weights.
    weights = torch.tensor(np.random.randn(11, 1), dtype=torch.float32).to(device)
    evaluate(weights, train_features, train_labels)
    weights.requires_grad_()

    # training using gradient descending
    trained_weights = gradient_desc(train_features, train_labels, weights, 0.002, 5000, -1)
    # de-normalize the weights
    trained_weights = de_normalise(trained_weights, mean, std)

    # evaluate the results
    print(evaluate(trained_weights, validate_features, train_labels))
    print(evaluate(trained_weights, validate_features, train_labels, mode="confusion_matrix"))
    print(trained_weights)


if __name__ == "__main__":
    # TOBE DONE:
    # anti overfit
    # sampling method
    main(False)
