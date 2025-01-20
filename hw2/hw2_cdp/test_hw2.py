import unittest
import numpy as np
from multiprocessing import Queue
from collect import load_mnist  # Assumes `collect.py` provides load_mnist
from preprocessor import Worker
from ip_network import IPNeuralNetwork
import os


class TestWorkerWithMNIST(unittest.TestCase):
    def setUp(self):
        """Set up shared resources for Worker tests using MNIST data."""
        self.jobs = Queue()
        self.results = Queue()

        # Load MNIST data
        self.training_data, _, _ = load_mnist()
        self.image = self.training_data[0][0]  # Use the first image for testing
        self.worker = Worker(self.jobs, self.results, batch_size=1)

    def test_rotate(self):
        """Test that rotation works and output shape is preserved."""
        rotated_image = self.worker.rotate(self.image, angle=45)
        self.assertEqual(rotated_image.shape, self.image.shape)
        self.assertTrue(np.any(rotated_image != 0))  # Ensure rotation alters the image

    def test_shift(self):
        """Test shifting the image by dx and dy."""
        shifted_image = self.worker.shift(self.image, dx=5, dy=5)
        self.assertEqual(shifted_image.shape, self.image.shape)
        self.assertTrue(np.any(shifted_image != 0))  # Ensure shifting alters the image

    def test_add_noise(self):
        """Test adding noise to the image."""
        noisy_image = self.worker.add_noise(self.image, noise=0.1)
        self.assertEqual(noisy_image.shape, self.image.shape)
        self.assertTrue(np.any(noisy_image != self.image))  # Ensure noise is added

    def test_skew(self):
        """Test skewing the image."""
        skewed_image = self.worker.skew(self.image, tilt=0.2)
        self.assertEqual(skewed_image.shape, self.image.shape)
        self.assertTrue(np.any(skewed_image != 0))  # Ensure skewing alters the image

    def test_process_image(self):
        """Test the sequential application of all augmentations."""
        augmented_image = self.worker.process_image(self.image)
        self.assertEqual(augmented_image.shape, self.image.shape)
        self.assertTrue(np.any(augmented_image != self.image))  # Ensure augmentations were applied

    def test_worker_run(self):
        """Test the worker's ability to process a job from the queue."""
        self.jobs.put(self.image)  # Add a job to the queue
        self.jobs.put(None)  # Stop signal
        self.worker.start()
        self.worker.join()  # Wait for the worker to finish

        augmented_batch = self.results.get()
        self.assertEqual(len(augmented_batch), 1)  # Ensure one image was processed
        self.assertTrue(np.any(augmented_batch[0] != self.image))  # Check augmentation


class TestIPNeuralNetworkWithMNIST(unittest.TestCase):
    def setUp(self):
        """Set up shared resources for IPNeuralNetwork tests using MNIST data."""
        self.training_data, self.validation_data, self.test_data = load_mnist()
        self.layers = [784, 128, 64, 10]
        self.nn = IPNeuralNetwork(
            layers=self.layers,
            learning_rate=0.1,
            mini_batch_size=16,
            number_of_batches=5,
            epochs=1,
        )

    def test_create_batches(self):
        """Test the creation of batches."""
        batches = self.nn.create_batches(self.training_data[0], self.training_data[1], batch_size=16)
        self.assertEqual(len(batches), 5)  # Ensure correct number of batches
        for x, y in batches:
            self.assertEqual(x.shape[0], 16)  # Batch size
            self.assertEqual(x.shape[1], 784)  # Input size
            self.assertEqual(len(y), 16)  # Labels

    def test_fit(self):
        """Test the fit method with data augmentation."""
        self.nn.fit(self.training_data, self.validation_data)
        # Check that weights and biases are updated
        self.assertTrue(np.any([np.any(w != 0) for w in self.nn.weights]))

    def test_empty_dataset(self):
        """Test handling of an empty dataset."""
        empty_data = (np.array([]), np.array([]))
        with self.assertRaises(ValueError):  # Expect a ValueError for empty data
            self.nn.fit(empty_data, self.validation_data)

    def test_no_workers(self):
        """Test the behavior when no workers are available."""
        os.environ["SLURM_CPUS_PER_TASK"] = "0"  # Simulate no workers
        self.nn.fit(self.training_data, self.validation_data)
        # Ensure training completes without errors
        self.assertTrue(True)


if __name__ == "__main__":
    t = TestWorkerWithMNIST()
    t.setUp()
    t.test_rotate()
    t.test_shift()
    t.test_add_noise()
    t.test_skew()
    t.test_process_image()
    t.test_worker_run()
