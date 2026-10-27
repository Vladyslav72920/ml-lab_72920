import numpy as np

INIT_STD = 0.1
def init_params(W, std=INIT_STD, seed=0):
    r = np.random.default_rng(seed)
    return {
        "W1": r.normal(0, std, (64, W)), "b1": np.zeros(W),
        "W2": r.normal(0, std, (W, W)),  "b2": np.zeros(W),
        "W3": r.normal(0, std, (W, 10)), "b3": np.zeros(10),
    }


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)   
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(X, p):
    Z1 = X @ p["W1"] + p["b1"];   A1 = np.maximum(0, Z1)
    Z2 = A1 @ p["W2"] + p["b2"];  A2 = np.maximum(0, Z2)
    Z3 = A2 @ p["W3"] + p["b3"];  P = softmax(Z3)
    return P, (X, Z1, A1, Z2, A2, P)


def loss_fn(P, Y):
    return -np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1))


def backward(Y, p, cache):
    """
    N = batch size
      dZ3 = (P - Y) / N
      dW3 = A2^T dZ3,  db3 = sum(dZ3)
      dZ2 = (dZ3 W3^T) * 1[Z2>0],  dW2 = A1^T dZ2,  db2 = sum(dZ2)
      dZ1 = (dZ2 W2^T) * 1[Z1>0],  dW1 = X^T dZ1,   db1 = sum(dZ1)
    """
    X, Z1, A1, Z2, A2, P = cache
    N = X.shape[0]
    g = {}
    dZ3 = (P - Y) / N
    g["W3"] = A2.T @ dZ3
    g["b3"] = dZ3.sum(axis=0)
    dZ2 = (dZ3 @ p["W3"].T) * (Z2 > 0)
    g["W2"] = A1.T @ dZ2
    g["b2"] = dZ2.sum(axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
    g["W1"] = X.T @ dZ1
    g["b1"] = dZ1.sum(axis=0)
    return g


def accuracy(p, X, y):
    return np.mean(forward(X, p)[0].argmax(axis=1) == y)