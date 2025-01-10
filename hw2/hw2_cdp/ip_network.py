#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from network import *
from multiprocessing import JoinableQueue, Queue, cpu_count
from preprocessor import Worker
import os

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        jobs = JoinableQueue()
        results = Queue()

        workers = [Worker(jobs, results) for _ in range(num_workers)]  # _ is not important
        for worker in workers:
            worker.start()

        data, labels = training_data
        mini_batches = self.create_batches(data, labels, self.mini_batch_size)
        for batch in mini_batches:
            jobs.put(batch)

        # 2. Set jobs
        jobs.join()

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        
        # 3. Stop Workers
        for worker in workers:
            worker.terminate()

        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
        Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        original_batches = super().create_batches(data, labels, batch_size)
        augmented_batches = []

        for images, lbls in original_batches:
            worker = Worker(jobs=None, result=None)  # Koren: Again, result or results?
            # Koren: worker was created just because the syntax of process_image obligates using a worker
            #   Is it violating the last line of page 5 in the homework?
            augmented_images = [Worker.process_image(worker, image) for image in images]
            augmented_batches.append((augmented_images, lbls))

        return augmented_batches
