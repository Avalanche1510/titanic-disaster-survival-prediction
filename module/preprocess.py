import pandas as pd
import torch
import numpy as np


def train_set():
    df = pd.read_csv("../titanic-disaster-survival-prediction/dataset/train.csv")
    # these column was removed as no obvious relationship to the label or too much missing data
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    # remove the samples without Age or Embarked
    df = df.dropna(subset=["Age", "Embarked"])
    # do one hot encoding for Embarked and Sex
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
    # return features and labels separately
    return df.iloc[:, 1:], df.iloc[:, 0]


def test_set():
    df_features = pd.read_csv("../titanic-disaster-survival-prediction/dataset/test.csv")
    # these column was removed as no obvious relationship to the label or too much missing data
    df_features = df_features.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    # remove the samples without Age or Embarked
    df_features = df_features.dropna(subset=["Age", "Embarked"])
    # do one hot encoding for Embarked and Sex
    df_features = pd.get_dummies(df_features, columns=["Sex", "Embarked"], dtype=int)

    df_labels = pd.read_csv("../titanic-disaster-survival-prediction/dataset/gender_submission.csv")
    df_labels = df_labels.drop(columns=["PassengerId"])

    return df_features, df_labels


def to_tensor(option="train"):
    if option == "train":
        features, labels = train_set()
        return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
    elif option == "test":
        features, labels = test_set()
        return torch.tensor(np.array(features), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
    else:
        return None


def main(option="train"):
    pd.set_option('display.max_columns', None)
    features, labels = to_tensor(option)
    print(features)
    print(features.shape)
    print(labels)
    print(labels.shape)


if __name__ == "__main__":
    main("test")
