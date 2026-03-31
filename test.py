import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from module.preprocess import *
from module.timer import Timer
from module.normalization import *
from module.evaluation import *


def gradient_desc(features, labels, weights, lr, epoch, batch_scale=-1, progress=True):
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
        if e % int(epoch / 16) == 0 and progress:
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


def sampling(df_features, df_labels, seed, method="random_frac", frac=0.8):
    if method == "random_frac":
        train_features = df_features.sample(frac=frac, random_state=seed)
        validation_features = df_features.drop(train_features.index)
        train_labels = df_labels.loc[train_features.index]
        validation_labels = df_labels.loc[validation_features.index]
        return train_features, train_labels, validation_features, validation_labels
    elif method == "stratify":
        train_neg = df_labels.loc[df_labels["Survived"] == 0].sample(frac=frac, random_state=seed)
        train_pos = df_labels.loc[df_labels["Survived"] == 1].sample(frac=frac, random_state=seed)
        # combine both positive and negative samples and refresh the order of samples
        train_labels = pd.concat([train_neg, train_pos]).sample(frac=1, random_state=seed)
        train_features = df_features.loc[train_labels.index]

        validation_labels = df_labels.drop(train_labels.index)
        validation_features = df_features.loc[validation_labels.index]
        return train_features, train_labels, validation_features, validation_labels
    else:
        return None


def main(gpu=False):
    seed = 42
    np.random.seed(seed)

    # initialise train and test datasets
    df_features, df_labels = train_set()
    df_labels = pd.DataFrame(df_labels)
    test_features = to_tensor(test_set())

    # sampling method apply to the data to separate train set and validation set
    train_features, train_labels, validation_features, validation_labels = sampling(df_features, df_labels, seed,
                                                                                    method="stratify", frac=0.8)



if __name__ == "__main__":
    # TOBE DONE:
    # sampling method
    main(False)
