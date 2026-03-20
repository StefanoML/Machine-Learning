import numpy as np
import matplotlib.pyplot as plt

theta = 0.0
learning_rate = 0.9
epochs = 100

def loss(theta):
    return (theta - 5) ** 2

def gradient(theta):
    return 2 * (theta -5) # derivative of (theta -5)**2 is 2*(theta-5)


for epoch in range(epochs):
    grad = gradient(theta)
    theta = (theta-learning_rate*grad)
    print(f"Epoch{epoch+1}: theta = {theta:.4f}, loss = {loss(theta):.4f}")

    if abs(grad) < 0.001:
        print (f"Converged at epoch {epoch+1}")
        break

thetas = []
losses = []
theta = 0.0

for epoch in range(100):
    grad = gradient(theta)
    theta = theta - learning_rate * grad
    thetas.append(theta)
    losses.append(loss(theta))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Gradient Descent - Loss over time")
plt.show()
