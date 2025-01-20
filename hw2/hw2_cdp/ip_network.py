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
from my_queue import MyQueue
from preprocessor import Worker, EndProcess, GenerateBatch
from network import *


class IPNeuralNetwork(NeuralNetwork):
    """
    Neural network that supports distributed data augmentation.
    """

    def __init__(self, layers, lr=0.1, batch_size=32, number_of_batches=20, epochs=10):
        super().__init__(layers, lr, batch_size, number_of_batches, epochs)
        self.jobs = JoinableQueue()
        self.results = MyQueue()
        self.processor_count = int(os.environ['SLURM_CPUS_PER_TASK'])

    def fit(self, training_data, validation_data=None):
        # Start workers
        processors = [
            Worker(self.jobs, self.results, training_data, self.mini_batch_size)
            for _ in range(self.processor_count)
        ]
        for proc in processors:
            proc.start()

        # Populate the task queue
        for _ in range(self.number_of_batches * self.epochs):
            self.jobs.put(GenerateBatch)

        # Call parent fit (training loop)
        super().fit(training_data, validation_data)

        # Send termination signal to workers
        for _ in processors:
            self.jobs.put(EndProcess)

        # Wait for all tasks to complete
        self.jobs.join()

        # Join workers
        for proc in processors:
            proc.join()

    def create_batches(self, data, labels, batch_size):
        """Generate augmented batches from the worker results queue."""
        augmented_batches = []
        for _ in range(self.number_of_batches):
            batch_images, batch_labels = self.results.get()
            # Ensure proper types for `batch_images` and `batch_labels`
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            augmented_batches.append((batch_images, batch_labels))
        return augmented_batches
