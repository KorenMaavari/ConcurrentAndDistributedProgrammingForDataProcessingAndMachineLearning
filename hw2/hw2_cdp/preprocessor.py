#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import random
import multiprocessing
from scipy.ndimage import rotate as scipy_rotate, shift as scipy_shift
import numpy as np


class Worker(multiprocessing.Process):
    """
    A parallel data augmentation worker that processes image batches.
    """

    def __init__(self, jobs, results, batch_size, training_data):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs #they don't have actual data inside that we'll use, we need them to manage the batch creation
        self.results = results #we will put here all the batches we will generate
        self.batch_size = batch_size
        self.training_data = training_data

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image

        Return
        ------
        An numpy array of same shape
        '''
        return scipy_rotate(image.reshape(28, 28), angle, reshape=False, mode='constant').flatten()

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis

        Return
        ------
        An numpy array of same shape
        '''
        return scipy_shift(image.reshape(28, 28), [dy, dx], mode='constant').flatten()

    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the
        range [-noise, noise] and added to it.

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        noise_array = np.random.uniform(-noise, noise, image.shape)
        noisy_image = image + noise_array
        return np.clip(noisy_image, 0, 1)

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        reshaped_img = image.reshape(28, 28)
        skewed_img = np.zeros_like(reshaped_img)
        for row in range(28):
            for col in range(28):
                skew_col = int(col + row * tilt)
                if 0 <= skew_col < 28:
                    skewed_img[row][col] = reshaped_img[row][skew_col]
        return skewed_img.flatten()

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        copied_image = np.copy(image)
        transformations = [
            (self.shift, [random.randint(-2, 2), random.randint(-2, 2)]),
            (self.rotate, [random.randint(-10, 10)]),
            (self.add_noise, [random.uniform(0.0, 0.15)]),
            (self.skew, [random.uniform(-0.1, 0.1)]),
        ]
        for func, params in transformations:
            copied_image = func(copied_image, *params)
        return copied_image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        # parallelized:
        # 1. compute batch indices, and get the data and labels in that indices (as it was in create_batches, but now every worker
        #     creates a batch, so it will be computed here)
        # 2. preform augmentation.
        # 3. put the result in the output queue

        while True:
            job = self.jobs.get()
            if job is None:  # Poison pill
                self.jobs.task_done()
                break

            data, labels = self.training_data
            indexes = random.sample(range(0, data.shape[0]), self.batch_size)
            batch_data = data[indexes]
            batch_labels = labels[indexes]
            aug_res = np.array([self.process_image(img) for img in batch_data])
            self.jobs.task_done()
            self.results.put((aug_res, batch_labels))