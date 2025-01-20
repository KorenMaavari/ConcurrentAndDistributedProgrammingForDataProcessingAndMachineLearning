#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from multiprocessing import Pipe, Lock, Value


class MyQueue(object):
    """
    A thread-safe queue for inter-process communication.
    """

    def __init__(self):
        """Initialize the queue and its synchronization mechanisms."""
        self.reader, self.writer = Pipe(duplex=False)
        self.lock = Lock()
        self.counter = Value('i', 0)

    def put(self, msg):
        """Put a message into the queue safely."""
        with self.lock:
            self.writer.send(msg)  # Send the message
        with self.counter.get_lock():
            self.counter.value += 1  # Increment the queue length counter

    def get(self):
        """Retrieve the next message from the queue safely."""
        msg = self.reader.recv()  # Receive the message
        with self.counter.get_lock():
            self.counter.value -= 1  # Decrement the queue length counter
        return msg

    def empty(self):
        """Check if the queue is empty."""
        with self.counter.get_lock():
            return self.counter.value == 0
