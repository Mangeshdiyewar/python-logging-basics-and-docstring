import  numpy as np
import os
import joblib

class Perceptron:
    def __init__(self, eta: float = None, epochs: int = None):
        self.weights = np.random.randn(3) * 1e-4  # small random weights
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initial weights before training: \n{self.weights}")
        self.eta = eta
        self.epochs = epochs

    def _z_outcome(self, inputs, weights):
        return np.dot(inputs, weights)

    def activation_function(self, z):
        return np.where(z > 0, 1, 0)

    def fit(self, x, y):
        self.x = x
        self.y = y

        x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]
        print(f"x with bias: \n{x_with_bias}")

        for epoch in range(self.epochs):
            print("--" * 10)
            print(f"for epoch >> {epoch}")
            print("--" * 10)

            z = self._z_outcome(x_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass: \n{y_hat}")

            self.error = self.y - y_hat
            print(f"error: \n{self.error}")

            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)
            print(f"updated weights after epochs: {epoch + 1}/{self.epochs}: \n{self.weights}")
            print("##" * 10)

    def predict(self, x):
        x_with_bias = np.c_[x, -np.ones((len(x), 1))]
        z = self._z_outcome(x_with_bias, self.weights)
        return self.activation_function(z)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"\n total loss:{total_loss}\n")

    def _create_dir_return_path(self, model_dir, filename):
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)

    def save(self, filename, model_dir=None):
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)

        else:
            model_file_path = self._create_dir_return_path("model", filename)
            joblib.dump(self, model_file_path)

    def load(self, filepath):
        return joblib.load(filepath)