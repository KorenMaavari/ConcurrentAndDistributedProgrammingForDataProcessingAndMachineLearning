#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from multiprocessing import Pipe, Lock


class MyQueue(object):

    def __init__(self):
        """ Initialize MyQueue and it's members.
        """
        """
        Initialize MyQueue with a lock for synchronization and a pipe for communication.

        Attributes:
        ----------
        lock : Lock
            Ensures thread-safe operations for multiple writers.
        pipe_reader : Connection
            End of the pipe for reading messages.
        pipe_writer : Connection
            End of the pipe for writing messages.
        """
        self.lock = Lock()  # Synchronize writers
        # duplex=False  ->  not bidirectional communication
        self.pipe_reader, self.pipe_writer = Pipe(duplex=False)  # Single reader, multiple writers

    def put(self, msg):
        """Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        """
        with self.lock:  # Lock ensures only one writer can send a message at a time
            self.pipe_writer.send(msg)

    def get(self):
        """Get the next message from queue (FIFO)

        Return
        ------
        An object
        """
        return self.pipe_reader.recv()  # Single reader, so no lock is needed

    def empty(self):
        """Get whether the queue is currently empty

        Return
        ------
        A boolean value
        """
        return not self.pipe_reader.poll()  # Check for data availability without blocking
