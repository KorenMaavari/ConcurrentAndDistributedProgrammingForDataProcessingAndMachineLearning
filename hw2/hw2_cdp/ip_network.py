#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import os
from multiprocessing import JoinableQueue
from multiprocessing import Queue
from preprocessor import Worker
from network import *


class IPNeuralNetwork(NeuralNetwork):
    """
    Neural network that supports distributed data augmentation.
    """

    def __init__(self, layers, learning_rate, mini_batch_size, number_of_batches, epochs):
        super().__init__(layers, learning_rate, mini_batch_size, number_of_batches, epochs)
        self.jobs = JoinableQueue()
        self.processor_count = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
        self.results = Queue()

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # first, create Workers
        workers = [
            Worker(self.jobs, self.results, self.mini_batch_size, training_data)
            for _ in range(self.processor_count)
        ]
        for worker in workers:
            worker.start()

        # now, call the parent class fit method
        super().fit(training_data, validation_data)

        # lastly, stop workers. we will give each worker poison pill to stop it.
        for _ in range(self.processor_count):
            self.jobs.put(None)

        self.jobs.join()

        for worker in workers:
            worker.join()

    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        batches = []
        for i in range(self.number_of_batches):
            # originally in Network, the batches were created here.
            # if we'd create here mini-batches: we'll have sequential operations and memory accesses to retrieve them + sequential augmentation!
            # we can speed it up by multiprocessing!
            self.jobs.put(i)

        # now, append all the batches that were created from the queue to the array.
        for i in range(0, self.number_of_batches):
            batches.append(self.results.get())

        return batches

