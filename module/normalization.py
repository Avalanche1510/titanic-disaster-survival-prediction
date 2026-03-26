import torch


def normalise(features, method="z-score"):
    if method == "z-score":
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
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

