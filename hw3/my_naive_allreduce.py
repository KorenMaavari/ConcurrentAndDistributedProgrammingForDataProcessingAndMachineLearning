import numpy as np
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # we want to collect data from all the processes.
    # so, we will send from each process it's send_buffer to all the other processes.
    # then, we will wait for all the data to be received.
    # when all the data is in place, calculate reduction in each process into recv_buffer.

    # allocate buffer for receiving values from all the process.
    all_values = [np.empty_like(send) for _ in range(size)]

    # send buffer to all other processes, and copy the send_buffer for the current process.
    for i in range(size):
        if i != rank:
            req = comm.Isend(send, dest=i)

    # receive back shared data from other processes
    for i in range(size):
        if i != rank:
            req = comm.Irecv(all_values[i], i)
            # wait for all the values!
            req.Wait()

    # reduce everyone
    compute = send
    for i in range(size):
        if i != rank:
            compute = np.vectorize(op)(all_values[i], compute)

    recv[:] = compute
    return recv


def sum(x,y):
    return x+y

def mul(x,y):
    return x*y


def do_allreduce():
    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    send_data = np.array([[1, 2, 3, 4, 5], [1 + rank, 2 + rank, 3 + rank, 4 + rank, 5 + rank], [10, -1, 2, rank, rank]])

    # sum
    recv_data = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, sum)

    print('====SUM==== \nrank: {} comm size: {}\n\n MY IMPL.\n send_data = {}\n recv_data = {}\n'.format(rank, size,
                                                                                                         send_data,
                                                                                                         recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.SUM)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    # mul
    recv_data = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, mul)
    print('====MUL==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank, size, send_data,
                                                                                            recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.PROD)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    # max- works only elementwize, not on np array
    recv_data = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, MPI.MAX)
    print('====MAX==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank, size, send_data,
                                                                                            recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.MAX)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    # min- works only elementwize, not on np array
    recv_data = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, MPI.MIN)
    print('====MIN==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank, size, send_data,
                                                                                            recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.MIN)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    MPI.Finalize()

if __name__ == '__main__':
    start = time()
    do_allreduce()
    stop = time()
    print(f'runtime: {stop-start}[s]')

