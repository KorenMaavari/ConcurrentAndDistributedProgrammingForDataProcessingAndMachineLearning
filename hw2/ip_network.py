#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import os
from multiprocessing import Queue
from my_queue import MyQueue
from preprocessor import Worker, EndProcess, GenerateBatch
from network import NeuralNetwork


class IPNeuralNetwork(NeuralNetwork):
    """
    Neural network that supports distributed data augmentation.
    """

    def __init__(self, layers, lr=0.1, batch_size=32, number_of_batches=20, epochs=10):
        super().__init__(layers, lr, batch_size, number_of_batches, epochs)
        self.task_queue = Queue()
        self.output_queue = MyQueue()
        self.processor_count = int(os.environ['SLURM_CPUS_PER_TASK'])

    def fit(self, training_data, validation_data=None):
        """Train the neural network with distributed workers."""
        # Launching parallel processors
        processors = [
            Worker(self.task_queue, self.output_queue, training_data, self.mini_batch_size)
            for _ in range(self.processor_count)
        ]
        for proc in processors:
            proc.start()

        # Populate task queue
        for _ in range(self.number_of_batches * self.epochs):
            self.task_queue.put(GenerateBatch)

        super().fit(training_data, validation_data)

        # Signal processors to terminate and wait for them to exit
        for _ in processors:
            self.task_queue.put(EndProcess)
        for proc in processors:
            proc.join()

    def create_batches(self, data, labels, batch_size):
        """Generate augmented batches from the worker results queue."""
        augmented_batches = []
        for _ in range(self.number_of_batches):
            augmented_batches.append(self.output_queue.get())
        return augmented_batches
