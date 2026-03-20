import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.random.randn(100)
y = 3 * X + 7 + np.random.randn(100) * 0.5

def predict ( X, theta_0, theta_1):
    return theta_0 + theta_1 * X

def mse_loss (X, y, theta_0, theta_1):
    predictions = predict(X,theta_0,theta_1)
    return np.mean((y-predictions)**2)

def gradient_theta_0(X, y, theta_0, theta_1):
    predictions = predict(X, theta_0, theta_1)
    return -2 * np.mean(y - predictions)

def gradient_theta_1(X, y, theta_0, theta_1):
    predictions = predict(X, theta_0, theta_1)
    return -2 * np.mean((y - predictions) * X)

theta_0 = 0.0
theta_1 = 0.0
learning_rate = 0.1
epochs = 100
losses = []

for epoch in range(epochs):
    grad_0 = gradient_theta_0(X, y, theta_0, theta_1)
    grad_1 = gradient_theta_1(X, y, theta_0, theta_1)
    
    theta_0 = theta_0 - learning_rate * grad_0
    theta_1 = theta_1 - learning_rate * grad_1
    
    losses.append(mse_loss(X, y, theta_0, theta_1))

print(f"Final theta_0: {theta_0:.4f}, expected ~7")
print(f"Final theta_1: {theta_1:.4f}, expected ~3")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Linear Regression - Gradient Descent")
plt.show()