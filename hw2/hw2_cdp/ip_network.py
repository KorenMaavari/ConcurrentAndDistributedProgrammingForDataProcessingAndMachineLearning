#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from network import *
from multiprocessing import JoinableQueue, Queue
from preprocessor import Worker
import os


class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        # Determine the number of available workers
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        jobs = JoinableQueue()
        results = Queue()
        workers = []

        # Start worker processes
        for _ in range(num_workers):
            worker = Worker(jobs, results, training_data, self.mini_batch_size)
            workers.append(worker)
            worker.start()

        # 2. Set jobs
        # Enqueue training data into the jobs queue for augmentation
        data, labels = training_data
        for batch in self.create_batches(data, labels, self.mini_batch_size):
            jobs.put(batch)

        # Signal workers to stop after all jobs are enqueued
        for _ in range(num_workers):
            jobs.put(None)

        # Collect augmented batches from the results queue
        jobs.join()
        augmented_batches = []
        while not results.empty():
            augmented_batches.extend(results.get())

        # Combine augmented data with original data
        augmented_training_data = (
            np.vstack([data] + [batch[0] for batch in augmented_batches]),
            np.concatenate([labels] + [batch[1] for batch in augmented_batches]),
        )

        # Call the parent class fit method with augmented data
        super().fit(augmented_training_data, validation_data)

        # 3. Stop Workers
        for _ in range(num_workers):
            jobs.put(None)
        for worker in workers:
            worker.join()



    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
        Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        # Create batches from the original dataset
        num_samples = len(data)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Shuffle the data for randomness

        batches = []
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_data = data[batch_indices]
            batch_labels = labels[batch_indices]
            batches.append((batch_data, batch_labels))

        return batches
