import math


class Neuron:
    """Implements a neuron and its basic operations."""
    activation_functions = {
        'linear': lambda x: x,
        'relu': lambda x: max(0, x),
        'step': lambda x: bool(x),
        'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
        'tanh': lambda x: math.tanh(x)
    }

    def __init__(self, activation='linear'):
        self.activation = self.activation_functions[activation]

    def __call__(self, inputs, weights, bias):
        """Computes output given inputs and weights.

        >>> neuron = Neuron()
        >>> inputs = [1, 2, 3]
        >>> weights = [-2, 4, 8]
        >>> bias = 5
        >>> neuron(inputs, weights, bias) == -1*2 + 2*4 + 3*8 + 5
        True
        """
        return self.activation(sum(i * w for i, w in zip(inputs, weights)) + bias)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
