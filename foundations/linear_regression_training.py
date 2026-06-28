import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64]) -> float:
        # note that N is just len(X)
        return 2 * X.T @ (model_prediction - ground_truth) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return (X @ weights).reshape(-1)

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = np.array(initial_weights).copy()

        for _ in range(num_iterations):
            model_prediction = self.get_model_prediction(X, weights)
            gradient = self.get_derivative(model_prediction, Y, len(X), X)
            weights = weights - self.learning_rate * gradient

        return np.round(weights, 5)