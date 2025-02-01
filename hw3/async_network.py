import mpi4py

from network import *

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters


    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with

        REMINDER: THIS CODE EXECUTES IN PARALLEL! EVERY WORKER EXECUTES IT!
        """
        # setting up the number of batches the worker should do every epoch
        # TODO: add your code

        # divide total number of batches between all the workers.
        num_batches_per_worker = self.number_of_batches // self.num_workers
        self.number_of_batches = num_batches_per_worker
        # needed to set up num_of_batches in order to create minibatches properly!
        # because num_of_batches minibatches are created when calling create_batches() method later in each epoch.

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters
                # TODO: add your code

                for layer in range(self.num_layers):
                    # we want to send the data to the master in charge of that layer.
                    # we want to specify a tag indicating the layer number.
                    # we will map the layers to tags as follows:
                    # for biases: tag[layer] = 2 * layer, for weights: tag[layer] = 2 * layer +1

                    tag_base = 2 * layer
                    master = layer % self.num_masters
                    self.comm.Isend(nabla_w[layer], master, tag_base + 1)
                    self.comm.Isend(nabla_b[layer], master, tag_base)


                # recieve new self.weight and self.biases values from masters
                # TODO: add your code

                for layer in range(self.num_layers):
                    tag_base = 2 * layer
                    master = layer % self.num_masters

                    req_w = self.comm.Irecv(self.weights[layer], master, tag_base + 1)
                    req_w.Wait()

                    req_b = self.comm.Irecv(self.biases[layer], master, tag_base)
                    req_b.Wait()



    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        self.number_of_batches = (self.number_of_batches // self.num_workers) * self.num_workers
        # because we changed num_of_batches in workers- we need them to match each other

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                # TODO: add your code

                # get the worker id to receive the message from
                status = MPI.Status()
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                worker = status.Get_source()

                for i, layer in enumerate(range(self.rank, self.num_layers, self.num_masters)):
                    tag_base = 2 * layer
                    req_w = self.comm.Irecv(nabla_w[i], worker, tag_base + 1)
                    req_w.Wait()

                    req_b = self.comm.Irecv(nabla_b[i], worker, tag_base)
                    req_b.Wait()

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                # TODO: add your code

                for layer in range(self.rank, self.num_layers, self.num_masters):
                    tag_base = 2 * layer
                    self.comm.Isend(self.weights[layer], worker, tag_base + 1)
                    self.comm.Isend(self.biases[layer], worker, tag_base)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        # TODO: add your code
        # if the master is not 0- send the calculations
        if self.rank != 0:
            for layer in range(self.rank, self.num_layers, self.num_masters):
                tag_base = 2 * layer
                self.comm.Isend(self.weights[layer], 0, tag_base + 1)
                self.comm.Isend(self.biases[layer], 0, tag_base)
        # if the master is 0- receive **from all other masters** the calculations and update
        if self.rank == 0:
            for other_master in range(1, self.num_masters):
                for layer in range(other_master, self.num_layers, self.num_masters):
                    tag_base = 2 * layer
                    recv_w_req = self.comm.Irecv(self.weights[layer], other_master, tag_base + 1)
                    recv_w_req.Wait()
                    recv_b_req = self.comm.Irecv(self.biases[layer], other_master, tag_base)
                    recv_b_req.Wait()
