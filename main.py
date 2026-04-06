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
            print(f"accuracy in train set: {evaluate(weights, features, labels)}")
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
    print("seed is: ", seed)
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
    elif method == "bootstrap":
        train_features = df_features.sample(frac=1.0, random_state=seed, replace=True)
        validation_features = df_features.drop(train_features.index)
        train_labels = df_labels.loc[train_features.index]
        validation_labels = df_labels.loc[validation_features.index]
        print(validation_labels.shape[0] / df_labels.shape[0] * 100, "% of the data is in the validation set.")
        return train_features, train_labels, validation_features, validation_labels
    else:
        return None


def main(gpu=False):

    # switch device: gpu/cpu
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"device using: {device}")

    # multiple runs of model to get average performance
    run = 50
    accuracy = []
    recall = []
    precision = []

    for run_index in range(run):
        # set different random seed by the number of runs
        np.random.seed(run_index)

        # initialise train and test datasets
        df_features, df_labels = train_set()
        df_labels = pd.DataFrame(df_labels)
        test_features = to_tensor(test_set())

        # sampling method apply to the data to separate train set and validation set
        train_features, train_labels, validation_features, validation_labels = sampling(df_features, df_labels, run_index**2, method="stratify", frac=0.8)

        train_features, train_labels = to_tensor(train_features, train_labels)
        validation_features, validation_labels = to_tensor(validation_features, validation_labels)
        print(train_features.shape, train_labels.shape)
        print(validation_features.shape, validation_labels.shape)

        if gpu:
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            validation_features = validation_features.to(device)
            validation_labels = validation_labels.to(device)
            test_features = test_features.to(device)

        # normalize using z-score
        train_features, mean, std = normalise(train_features, "z-score")

        # there's 10 weights for 10 features and 1 bias, in total 11 weights.
        weights = torch.tensor(np.random.randn(11, 1), dtype=torch.float32).to(device)
        evaluate(weights, train_features, train_labels)
        weights.requires_grad_()

        # print(train_labels.shape)
        # print((train_labels[train_labels == 0]).shape)
        # print((train_labels[train_labels==1]).shape)

        # training using gradient descending
        trained_weights = gradient_desc(train_features, train_labels, weights, 0.01, 3000, -1)
        # de-normalize the weights
        trained_weights = de_normalise(trained_weights, mean, std)

        # evaluate the results
        print(f"This is the run <{run_index+1}> !")
        accuracy.append(evaluate(trained_weights, validation_features, validation_labels, mode="accuracy"))
        precision.append(evaluate(trained_weights, validation_features, validation_labels, mode="precision"))
        recall.append(evaluate(trained_weights, validation_features, validation_labels, mode="recall"))
        print("accuracy (in validation set): ", accuracy[run_index])
        print("precision: ", precision[run_index])
        print("recall: ", recall[run_index])
        print("confusion matrix: ", evaluate(trained_weights, validation_features, validation_labels, mode="confusion_matrix"))
        print(trained_weights)
    print("\n\n\n")
    print("Final Accuracy: ", np.array(accuracy).mean())
    print("Final Precision: ", np.array(precision).mean())
    print("Final Recall: ", np.array(recall).mean())
    print("Accuracy std: ", np.array(accuracy).std())
    print("Precision std: ", np.array(precision).std())
    print("Recall std: ", np.array(recall).std())


"""
50 runs average results:

Random Frac frac=0.8 ~ B
Final Accuracy:  0.7885915492957746
Final Precision:  0.7543982334192415
Final Recall:  0.7067385303892931
Accuracy std:  0.03038762994966123
Precision std:  0.06294473136127064
Recall std:  0.05037237566592636

Stratify frac=0.8 ~ A
Final Accuracy:  0.7977622377622379
Final Precision:  0.7748982857701491
Final Recall:  0.7127586206896551
Accuracy std:  0.035072283071611546
Precision std:  0.05754695391210016
Recall std:  0.0646595400823599

Bootstrap ~ B
Final Accuracy:  0.7865250134868224
Final Precision:  0.7583520968498099
Final Recall:  0.7020016478374325
Accuracy std:  0.02057853411986606
Precision std:  0.03920533165682601
Recall std:  0.04753781380151119
"""
if __name__ == "__main__":
    # TOBE DONE:
    # sampling method
    main(False)
