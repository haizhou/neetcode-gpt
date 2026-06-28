import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        z0 = x @ W1.T + b1
        z1 = np.maximum(0, z0)
        z2 = z1 @ W2.T + b2
        l = np.mean(np.square(z2 - y_true))
        dl_dz2 = 2 * (z2 - y_true) / len(y_true)
        dl_dw2 = np.outer(dl_dz2, z1)
        dl_db2 = dl_dz2
        dl_dz1 = dl_dz2 * W2
        dl_dz0 = dl_dz1 * (z0 > 0).astype(float)
        dl_db1 = dl_dz0.reshape(-1)
        dl_dw1 = x * dl_dz0.reshape(-1,1)
        return {
            'loss': round(float(l), 4),
            'dW1': np.round(dl_dw1, 4).tolist(),
            'db1': np.round(dl_db1, 4).tolist(),
            'dW2': np.round(dl_dw2, 4).tolist(),
            'db2': np.round(dl_db2, 4).tolist(),
        }

