import numpy as np
import matplotlib.pyplot as plt

def kernel_rbf(x, gamma):
    NoD = len(x)
    K = np.zeros((NoD, NoD))
    for i in range(NoD):
        for j in range(NoD):
            K[i, j] = np.exp(-gamma * (x[i] - x[j]) ** 2)
    return K

def kalman_filter(a_k, b_k, x, P, Q, R):
    a_k = a_k.reshape(-1, 1)
    b_k = b_k.reshape(-1, 1)
    # Prediction Phase
    x_pred = x  # Predicted state (No process model here)
    P_pred = P + Q  # Predicted covariance

    # Correction Phase
    K = (P_pred @ a_k) / (a_k.T @ P_pred @ a_k + R)  # Compute Kalman Gain
    x = x_pred + K @ (b_k - a_k.T @ x_pred)  # State Update
    P = (np.eye(len(K)) - K @ a_k.T) @ P_pred  # Covariance Update
    return x, K, P

# Input and Output definition
xtrain = np.arange(1, 20.1, 0.1).reshape(-1, 1)
NoD = len(xtrain)

# ytrain = np.sin(xtrain) + 2 * np.sin(2 * xtrain) + 0.5 * np.random.randn(NoD, 1)
ytrain = 0.01 * xtrain ** 2 + 0.1 * np.exp(-xtrain) + np.sin(xtrain) + 0.1 * np.random.randn(NoD, 1)

# Train RLS-SVM
C = 100  # Over Fitting Param (C=100)
gamma = 5e-1  # RBF param, equal to 1/2sigma^2 (g=1e-2)
kernelSelect = 'rbf'
K = kernel_rbf(xtrain, gamma)
A = np.vstack([np.zeros(NoD), np.ones(NoD)]).T
A = np.hstack([A, K + 1 / C * np.eye(NoD)])
b = np.vstack([np.zeros((1, 1)), ytrain])

# Kalman-Tuned RLS-SVR
x_kalman = np.random.rand(A.shape[1], 1)  # Random kalman state vector

# Kalman Filter Parameters
P = 1e1 * np.eye(A.shape[1])
Q = 1e-5 * np.eye(A.shape[1])  # Process noise covariance
R = 1e-2  # Measurement noise covariance

# Prediction Stage
xpred = xtrain  # Prediction Input
ypred = np.zeros(NoD)
tmp = np.zeros(NoD)

plt.figure(figsize=(10, 6))
for k in range(NoD):
    x_kalman, K, P = kalman_filter(A[k, :], b[k, :], x_kalman, P, Q, R)
    # Bias and alpha LaGrange Multipliers
    bias = x_kalman[0]
    alpha = x_kalman[1:]

    for j in range(NoD):
        for i in range(NoD):
            tmp[i] = alpha[i] * np.exp(-gamma * (xpred[j] - xpred[i]) ** 2)
        ypred[j] = np.sum(tmp) + bias

    # Plot Result
    plt.clf()
    plt.plot(xtrain, ytrain, 'ko-', linewidth=2, label='Noisy Data')
    plt.plot(xpred, ypred, 'r.-', linewidth=2, label='Kalman-Tuned SVR')
    plt.axis([0, 21, -2, 5.5])
    plt.title('Kalman-Tuned SVR Nonlinear Regression')
    plt.legend()
    plt.draw()
    plt.pause(0.01)

plt.show()
