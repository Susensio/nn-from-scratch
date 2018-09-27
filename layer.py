from neuron import Neuron
import numpy as np


class Layer:
    """Implements a list of neurons and its basic operations."""
    activation_functions = {name: np.vectorize(func) for name, func in Neuron.activation_functions.items()}

    def __init__(self, size, activation='linear'):
        self.activation = self.activation_functions[activation]
        self.neurons = [Neuron(activation) for _ in range(size)]

    def __call__(self, inputs, weights, bias):
        """Computes a forward pass.

        >>> layer = Layer(3, 'sigmoid')
        >>> inputs = [1, 2]
        >>> weights = [[1, 1] , [2, 2], [3, 3]]
        >>> bias = 5
        >>> output = layer(inputs, weights, bias)

        >>> all(layer.activation(np.array(weights) @ inputs + bias) == np.array(output))
        True
        """
        return [neuron(inputs, weights[i], bias) for i, neuron in enumerate(self.neurons)]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
