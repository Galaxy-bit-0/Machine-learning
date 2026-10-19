import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()

X = np.array(iris.data)
y = iris.target
y = (y == 2).astype(int)  #type conversion

theta = np.zeros(X.shape[1])
bias = 0.0
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
m = X_train.shape[0]

X = X_train
y = y_train
alpha = 0.01

def predict(theta,bias,X):
    sigmoid = 1 / (1 + np.exp(-(X @ theta + bias)))

    return sigmoid

def cost_function(sigmoid, m, y):
    # Keeps probabilities bounded between 0.000000000000001 and 0.999999999999999

    sigmoid = np.clip(sigmoid, 1e-15, 1 - 1e-15)
    loss = -1 / m * (np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid)))

    return loss

def gradient(sigmoid, y, theta,bias, alpha):
    
    grad_theta = (1 / m) * (X.T @ (sigmoid - y))
    grad_bias = (1 / m) * np.sum(sigmoid - y)
    theta = theta - alpha * grad_theta
    bias = bias - alpha * grad_bias
    return theta,bias

#algorithm gradient desecnt training
costs = []
for epoch in range(1000):
    predictions = predict(theta, bias, X_train)
    cost = cost_function(predictions, m, y_train)
    costs.append(cost)
    theta, bias = gradient(predictions, y_train, theta, bias, alpha)

#Accuracy
n = X_test.shape[0] #number of test samples
test_probabilities = predict(theta,bias,X_test)
test_prediction = (test_probabilities >=0.5).astype(int)
"""accuracy = 1/n*(np.sum(test_prediction == y_test))""" # is same as np.mean()
accuracy = (1/n) * np.sum(test_prediction == y_test)
print(f"accuracy of the models is {accuracy*100:.2f}%")

plt.plot(costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training Cost over Epochs")
plt.show()


# Train on just 2 features (petal length, petal width) purely for visualization
X_vis = X_train[:, 2:4]
theta_vis = np.zeros(2)
bias_vis = 0.0
m_vis = X_vis.shape[0]

for epoch in range(1000):
    preds = 1 / (1 + np.exp(-(X_vis @ theta_vis + bias_vis)))
    error = preds - y_train
    grad_theta = (1/m_vis) * (X_vis.T @ error)
    grad_bias = (1/m_vis) * np.sum(error)
    theta_vis -= alpha * grad_theta
    bias_vis -= alpha * grad_bias

# Decision boundary plot using YOUR model (2-feature version)
x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                      np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = (1 / (1 + np.exp(-(grid @ theta_vis + bias_vis))) >= 0.5).astype(int)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, cmap="coolwarm", edgecolor="k")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Decision Boundary — Your Model")
plt.show()