import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0        
y = digits.target
Y = np.eye(10)[y]               

X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
    X, Y, y, test_size=0.3, stratify=y, random_state=42)

if __name__ == "__main__":
    print("Train:", X_tr.shape, "Test:", X_te.shape)