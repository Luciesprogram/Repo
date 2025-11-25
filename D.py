Question 3: How do we implement this generalized linear formula in core Python?

def _compute_z(theta_0, theta_weights, x_sample):

if len(theta_weights) != len(x_sample):

	raise ValueError("""Mismatch in length of weights and features\n

		Or, Mismatch in the length of theta_1 and X""")

z = theta_0

for i in range(len(theta_weights)):

	z += theta_weights[i] * x_sample[i]

return z



Question 8: Show the Python implementation of the Sigmoid function and the new hypothesis.



import math

def _sigmoid(z):

"""

The Sigmoid activation function.

Includes guards for numerical stability to prevent overflow.

"""

if z > 700: # e^(-700) is effectively 0

	return 1.0

elif z < -700: # e^(700) is effectively infinity

	return 0.0

else:

	return 1.0 / (1.0 + math.exp(-z))





# The new hypothesis function simply combines the linear calculation with the sigmoid activation:

def _predict_probability(theta_0, theta_weights, x_sample):



	z = _compute_z(theta_0, theta_weights, x_sample)

	return _sigmoid(z)





13.  Provide the complete, commented Python code for the CoreLogisticRegression class.

import math

class CoreLogisticRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):

        """Initializes the model's hyperparameters."""

        self.learning_rate = learning_rate

        self.n_iterations = n_iterations

        self.theta_0 = 0.0

        self.theta_weights = []

        self.cost_history = []

    def _sigmoid(self, z):

        """Computes the sigmoid function with numerical stability."""

        if z > 700: return 1.0

        elif z < -700: return 0.0

        else: return 1.0 / (1.0 + math.exp(-z))

    def _compute_z(self, x_sample):

        """Computes the linear combination of inputs and weights."""

        z = self.theta_0

        for i in range(len(self.theta_weights)):

            z += self.theta_weights[i] * x_sample[i]

        return z

    def _predict_probability(self, x_sample):

        """Makes a probability prediction for a single sample."""

        z = self._compute_z(x_sample)

        return self._sigmoid(z)

    

    def _compute_cost(self, y_true, y_pred_probs):

        """Computes the binary cross-entropy (log loss)."""

        m = len(y_true)

        if m == 0: return 0.0

        total_cost, epsilon = 0.0, 1e-9

        for i in range(m):

            h = max(epsilon, min(1.0 - epsilon, y_pred_probs[i])) # Clipping

            cost_sample = -y_true[i] * math.log(h) - (1 - y_true[i]) * math.log(1 - h)

            total_cost += cost_sample

        return total_cost / m

    

    def _compute_gradients(self, X_data, y_true, y_pred_probs):

        """Computes the gradients of the cost function."""

        m, n_features = len(y_true), len(self.theta_weights)

        grad_theta_0 = 0.0

        grad_theta_weights = [0.0] * n_features

        for i in range(m):

            error = y_pred_probs[i] - y_true[i]

            grad_theta_0 += error

            for j in range(n_features):

                grad_theta_weights[j] += error * X_data[i][j]

        grad_theta_0 /= m

        for j in range(n_features): grad_theta_weights[j] /= m

        return grad_theta_0, grad_theta_weights

    def fit(self, X_data, y_data, verbose=True):

        """Trains the model using batch gradient descent."""

        n_features = len(X_data[0])

        self.theta_0 = 0.0

        self.theta_weights = [0.0] * n_features

        self.cost_history = []

        for i in range(self.n_iterations):

            y_pred_probs = [self._predict_probability(x) for x in X_data]

            cost = self._compute_cost(y_data, y_pred_probs)

            self.cost_history.append(cost)

            grad_theta_0, grad_theta_weights = self._compute_gradients(X_data, y_data, y_pred_probs)

            self.theta_0 -= self.learning_rate * grad_theta_0

            for j in range(n_features):

                self.theta_weights[j] -= self.learning_rate * grad_theta_weights[j]

            if verbose and i % (self.n_iterations // 10) == 0:

                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict_proba(self, X_data):

        """Predicts probabilities for new data."""

        return [self._predict_probability(x) for x in X_data]

    def predict(self, X_data, threshold=0.5):



        probabilities = self.predict_proba(X_data)

        return [1 if prob >= threshold else 0 for prob in probabilities]





Question 14: How do we test this model? Provide the setup for a sample experiment.



if __name__ == "__main__":

print("--- Testing CoreLogisticRegression ---")

# 1. Create a simple dataset

X_train = [[1.0], [1.5], [2.0], [2.5], [4.5], [5.0], [5.5], [6.0]]

y_train = [0, 0, 0, 0, 1, 1, 1, 1] # Students who studied > 4 hours passed

# 2. Initialize and train the model

model = CoreLogisticRegression(learning_rate=0.1, n_iterations=5000)

print("Starting training...")

model.fit(X_train, y_train)

print("Training complete.")

# 3. Print the final learned parameters

print(f"\nFinal Bias (theta_0): {model.theta_0:.4f}")

print(f"Final Weights (theta_1): {model.theta_weights[0]:.4f}")

# 4. Make predictions on new, unseen data

X_test = [[0.5], [3.0], [3.5], [7.0]]

probs = model.predict_proba(X_test)

labels = model.predict(X_test)

print("\n--- Test Results ---")

for i in range(len(X_test)):

	print(f"Input: {X_test[i][0]} hours | "

	f"Prob(Pass): {probs[i]:.4f} | "

	f"Prediction: {labels[i]}")

