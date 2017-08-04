

def dx (f, x):
    return abs(0 - f(x))


def newtons_method (f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    print('Root is at: ' + str(x0))
    print('f(x) at root is: ' + str(f(x0)))


def function (x):
    return 3 * (x ** 3) - (x ** 2) - 3


def deriv_func (x):
    return 9 * (x ** 2) - (2 * x)


newtons_method(function, deriv_func, 5, 0.1)

y = lambda x: 9 * (x ** 2) - (2 * x)
print(y(9))
