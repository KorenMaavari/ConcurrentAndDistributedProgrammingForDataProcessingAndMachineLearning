#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import my_queue
import random
import multiprocessing
from scipy.ndimage import rotate as scipy_rotate, shift as scipy_shift
import numpy as np


# Markers for communication commands
class EndProcess:
    pass


class GenerateBatch:
    pass


class Worker(multiprocessing.Process):
    """
    A parallel data augmentation worker that processes image batches.
    """

    def __init__(self, task_queue: multiprocessing.Queue, output_queue: my_queue.MyQueue, dataset, batch_size: int):
        """
        Initialize the Worker with a task queue, result queue, dataset, and batch size.

        Parameters
        ----------
        task_queue : Queue
            Queue holding tasks for processing.
        output_queue : Queue
            Queue where processed results are stored.
        dataset : tuple
            A tuple containing (images, labels).
        batch_size : int
            Number of images per batch.
        """
        super().__init__()
        self.task_queue = task_queue
        self.output_queue = output_queue
        self.dataset = dataset
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        """Rotate the image by the specified angle."""
        return scipy_rotate(image.reshape(28, 28), angle, reshape=False).flatten()

    @staticmethod
    def shift(image, dx, dy):
        """Shift the image by the given pixel offsets."""
        return scipy_shift(image.reshape(28, 28), (dx, dy), cval=0).flatten()

    @staticmethod
    def add_noise(image, noise):
        """Add random noise within [-max_noise, max_noise] to the image."""
        actual_noise = np.random.uniform(-noise, noise, image.shape)
        noisy_img = np.clip(image + actual_noise, 0, 1)
        return noisy_img

    @staticmethod
    def skew(image, tilt):
        """Skew the image by a specified factor."""
        reshaped_img = image.reshape(28, 28)
        skewed_img = np.zeros_like(reshaped_img)
        for row in range(28):
            for col in range(28):
                skew_col = int(col + row * tilt)
                if 0 <= skew_col < 28:
                    skewed_img[row][col] = reshaped_img[row][skew_col]
        return skewed_img.flatten()

    def process_image(self, image):
        """Perform multiple augmentations in random order."""
        transformations = [
            (self.rotate, [random.randint(-5, 5)]),
            (self.shift, [random.randint(-3, 3), random.randint(-3, 3)]),
            (self.add_noise, [random.uniform(0.0, 0.02)]),
            (self.skew, [random.uniform(-0.1, 0.1)]),
        ]
        random.shuffle(transformations)
        for func, params in transformations:
            image = func(image, *params)
        return image

    def run(self):
        """Continuously process batches from the task queue."""
        images, labels = self.dataset
        while True:
            task = self.task_queue.get()
            if task is EndProcess:
                break
            indices = random.sample(range(len(images)), self.batch_size)
            augmented_images = [self.process_image(images[i]) for i in indices]
            self.output_queue.put((augmented_images, labels[indices]))
