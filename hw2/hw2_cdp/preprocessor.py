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
from scipy.ndimage import rotate as scipy_rotate, shift as scipy_shift

class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
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
        self.training_data = training_data
        self.batch_size = batch_size

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

        # Create a 2x2 identity matrix as the base transformation matrix.
        # Initially, this matrix performs no transformation.
        skew_matrix = np.eye(2)

        # Modify the [0, 1] element of the matrix to add a horizontal skew transformation.
        # tilt determines the strength and direction of the skew.
        skew_matrix[0, 1] = tilt

        # Generate a grid of pixel coordinates for the image.
        # np.indices((28, 28)) creates two 2D arrays:
        # - The first array contains the row indices of the image.
        # - The second array contains the column indices of the image.
        # Reshape these arrays into a single array of flattened coordinates.
        coords = np.indices((28, 28)).reshape(2, -1)

        # Apply the skew transformation to the pixel coordinates using matrix multiplication.
        # This shifts the x-coordinates based on the tilt value while keeping y-coordinates unchanged.
        # Round the resulting coordinates to integers and convert them to int type.
        new_coords = np.dot(skew_matrix, coords).round().astype(int)

        # Ensure all transformed coordinates remain within the bounds of the image (0 to 27).
        # This prevents accessing invalid indices outside the image dimensions.
        new_coords[0] = np.clip(new_coords[0], 0, 27)  # Clip row indices.
        new_coords[1] = np.clip(new_coords[1], 0, 27)  # Clip column indices.

        # Create a blank (zero-initialized) canvas with the same shape as the original image.
        # This will hold the pixel values of the skewed image.
        skewed_image = np.zeros_like(image.reshape(28, 28))

        # Map the pixel values from the original image to the new coordinates.
        # For each transformed coordinate, place the corresponding pixel value in the skewed image.
        skewed_image[new_coords[0], new_coords[1]] = image.reshape(28, 28)

        # Flatten the 2D skewed image into a 1D array to ensure compatibility with other parts of the pipeline.
        return skewed_image.flatten()

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
        angle = np.random.uniform(-30, 30)
        dx, dy = np.random.uniform(-3, 3, 2)
        noise = np.random.uniform(0, 0.2)
        tilt = np.random.uniform(-0.3, 0.3)

        image = self.rotate(image, angle)
        image = self.shift(image, dx, dy)
        image = self.add_noise(image, noise)
        image = self.skew(image, tilt)

        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            try:
                image = self.jobs.get()
                if image is None:  # Termination signal
                    break  # Because self.jobs.empty() == True
                augmented_image = self.process_image(image)
                self.result.put(augmented_image)
                self.jobs.task_done()
            except Exception as e:
                print(f"Error in Worker: {e}")
