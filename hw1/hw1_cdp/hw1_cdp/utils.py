import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.ndarray input data

     Returns
     -------
     np.ndarray
         sigmoid of the input x
     """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.ndarray input data

         Returns
         -------
         np.ndarray
             derivative of sigmoid of the input x

    """
    sig = sigmoid(x)
    return sig * (1 - sig)


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of xavier initialized np arrays weight matrices
    """
    return [xavier_initialization(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]  # Koren: Or range(len(sizes))


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays weight matrices
    """
    return [np.zeros((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]  # Koren: Maybe range of len(sizes)


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays bias matrices
    """

    # return [np.zeros(size) for size in sizes[1:]]
    return [np.zeros(sizes[i]) for i in range(1, len(sizes))]  # Koren: Maybe range(1, len(sizes) + 1)


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.ndarray of input data
         labels : np.ndarray of input labels
         batch_size : int size of batch

         Returns
         -------
         list of tuples of (data batch of batch_size, labels batch of batch_size)
    """

    n = len(data)  # n = data.shape[0]
    batches = [
        (data[i:i + batch_size], labels[i:i + batch_size])
        for i in range(0, n, batch_size)
    ]
    return batches


def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.ndarray of numbers
         list2 : np.ndarray of numbers

         Returns
         -------
         list of sum of each two elements by index
    """
    return [a + b for a, b in zip(list1, list2)]


def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
