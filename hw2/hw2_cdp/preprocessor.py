#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
import numpy as np
from scipy.ndimage import rotate, shift

class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        # Koren: Here there are 5 inputs, but in the homework it is with 3 inputs
        # Koren: Here and in page 4 of the homework the "result" input is without the letter s, but in page 3 of the
        #   homework it is "results" with the letter s
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
        self.jobs = jobs
        self.result = result

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
        return rotate(image.reshape(28, 28), angle, reshape=False, mode='constant', cval=0).flatten()

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
        return shift(image.reshape(28, 28), [dy, dx], mode='constant', cval=0).flatten()
    
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
        actual_noise = np.random.uniform(-noise, noise, image.shape)
        return np.clip(image + actual_noise, 0, 1)

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
        skewed = np.zeros_like(image.reshape(28, 28))
        for i in range(28):  # Iterate over each row
            offset = int(i * tilt)  # Round to the row index
            if 0 <= offset < 28:
                skewed[i, offset:] = image.reshape(28, 28)[i, :28-offset]
                # Koren: May be skewed[i, :] = image.reshape(28, 28)[i, offset:]
        return skewed.flatten()

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
        image = self.rotate(image, np.random.uniform(-30, 30))
        image = self.shift(image, np.random.randint(-5, 5), np.random.randint(-5, 5))
        image = self.add_noise(image, 0.2)
        image = self.skew(image, np.random.uniform(-0.3, 0.3))
        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while not self.jobs.empty():
            batch = self.jobs.get()
            augmented_batch = [(self.process_image(image), label) for image, label in batch]
            self.result.put(augmented_batch)  # Koren: Again, result or results?
            self.jobs.task_done()
