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

'''
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
plt.show()
'''


# perform stochastic gradient descent to learn the seperating hyperplane
def svm_sgd_plot (X, Y):
    # initialize SVM weight vector with zeros
    w = np.zeros(len(X[0]))
    # learning rate
    eta = 1
    # training iterations
    epochs = 100000
    # store misclassifications to plot how they change over time
    errors = []

    # training and gradient descent
    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            # misclassification
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * ( (X[i] * Y[i]) + (-2 * (1/epoch) * w) )
                error = 1
            else:
                w = w + eta * (-2 * (1/epoch) * w)
        errors.append(error)

    # plot the rate of classification errors during training
    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.show()

    return w


w = svm_sgd_plot(X, y)

for d, sample in enumerate(X):
    # plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)


# add test samples
plt.scatter(2, 2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4, 3, s=120, marker='+', linewidths=2, color='blue')

# print the hyperplane calculated 
x2 = [w[0], w[1], -w[1],  w[0]]
x3 = [w[0], w[1],  w[1], -w[0]]

x2x3 = np.array([x2, x3])
X, Y, U, V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X, Y, U, V, scale=1, color='blue')

plt.show()

