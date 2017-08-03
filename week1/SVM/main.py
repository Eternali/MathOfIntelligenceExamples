import numpy as np
from matplotlib import pyplot as plt

# input data - of the form [X value, Y value, Bias term]
X = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1]
])

# associated output labels
y = np.array([-1, -1, 1, 1, 1])

# plot these examples on a 2D graph
for d, sample in enumerate(X):
    # plot negative samples (d = 0, 1)
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # plot positive samples (d = 2,3,4)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# print a possible hyperplane, that is seperating the two classes. (this is a naive guess)
plt.plot([-2, 6], [6, 0.5])

