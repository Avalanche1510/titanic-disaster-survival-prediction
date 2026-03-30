import torch


def evaluate(weights, test_features, test_labels, mode="accuracy"):
    predict_labels = test_features @ weights[1:] + weights[0]
    predict_labels = torch.sigmoid(predict_labels)
    predict = predict_labels > 0.5
    real = test_labels > 0.5
    sample_num = predict.shape[0]
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for s in range(sample_num):
        if predict[s, 0]:
            if real[s, 0]:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if real[s, 0]:
                false_negative += 1
            else:
                true_negative += 1
    if mode == "confusion_matrix":
        return true_positive, false_positive, true_negative, false_negative
    elif mode == "recall":
        return true_positive / (true_positive + false_negative), true_negative / (true_negative + false_positive)
    elif mode == "accuracy":
        return (true_positive+true_negative)/sample_num, (false_positive+false_negative)/sample_num
    elif mode == "precision":
        return true_positive / (true_positive + false_positive), true_negative / (true_negative + false_negative)
    else:
        return None
