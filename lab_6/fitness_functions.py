__all__ = ('model1', 'model2', 'model3')


def model1(x, b):
    return b[0] * (1 - 1 / (1 + (b[1]*x)/2)**2)


def model2(x, b):
    return (b[0] + b[1]*x + b[2]*x**2) / (1 + b[3]*x + b[4]*x**2)


def model3(x, b):
    return (b[0] + b[1]*x + b[2]*x**2 + b[3]*x**3) / (1 + b[4]*x + b[5]*x**2 + b[6]*x**3)
