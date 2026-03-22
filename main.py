import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from preprocess import *
from timer import Timer


def main():
    train_features, train_labels = to_tensor("train")
    test_features, test_labels = to_tensor("test")

    print(train_features[0:9, :])
    print(test_features[0:9, :])


if __name__ == "__main__":
    main()
