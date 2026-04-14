import torch


def normalise(features, method="z-score"):
    # for the columns of one hot encoding, the mean is 0 and std is 1,
    # so there's a possibility of std being 0, which will cause error when normalising.
    # To avoid this, we add a small value to std when it's 0.
    if method == "z-score":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        if torch.any(std == 0):
            std = std + 1e-8
        features = (features - mean)/std
        return features, mean, std
    else:
        return None


def de_normalise(weights, mean=None, std=None, method="z-score"):
    if method == "z-score":
        with torch.no_grad():
            weights[0, 0] -= (weights[1:, :] * mean.T / std.T).sum()
            weights[1:, :] /= std.T
        return weights
    else:
        return None

