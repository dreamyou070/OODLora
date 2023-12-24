import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
x_hat = np.array([[1, 5, 2], [8, 7, 1]])
mse = np.square(x - x_hat) ** 0.5
binary = np.where(mse > 0.5, 1, 0)
print(binary)