import numpy as np
import pandas as pd
import torch

a = torch.randn(1000, 1)
a = pd.DataFrame(a)
print(a, "\n")
b = a.sample(frac=1.0, random_state=49, replace=True)
print(b, "\n")
c = a.drop(b.index)
print(c, "\n")
print(c.shape[0] / a.shape[0] * 100, "% of the data is in the validation set.")