import numpy as np

from network import *
from my_ring_allreduce import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class SynchronicNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):

        MPI.Init()
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for epoch in range(self.epochs):

            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size // size)

            for x, y in mini_batches:
                # doing props
                self.forward_prop(x)
                my_nabla_b, my_nabla_w = self.back_prop(y) # my_nabla is a list of numpy array of length NN_num_layers!

                # summing all my_nabla_b and my_nabla_w to nabla_w and nabla_b
                # TODO: add your code
                # use global computation + all reduce: we want to access the reduced data on all processors

                # first - allocate buffers
                nabla_w = [np.zeros_like(i) for i in my_nabla_w]
                nabla_b = [np.zeros_like(i) for i in my_nabla_b]

                # now- use AllReduce!
                # notice: The sum: nabla[pos, idx] = sum_from_all(my_nabla[pos, idx])
                for src, dst in zip (my_nabla_w, nabla_w):
                    comm.Allreduce(src, dst, op=MPI.SUM)

                for src, dst in zip(my_nabla_b, nabla_b):
                    comm.Allreduce(src, dst, op=MPI.SUM)

                # ToDo: average?

                # end of our code

                # calculate work
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
