import numpy as np
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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
    num_processes = comm.Get_size()

    # first, we want to pass values to all the processes in ring and [op] them up.
    # each iteration, a process sends a message (contains a chunk of the data) and receives one,
    # and then it needs to preform the op on the values it got with its array.
    # after preforming n-1 [op]s, we will need to distribute them.

    # prepare for each process the data it will send in chunks
    chunks = np.array_split(send, num_processes)

    # prepare for each process buffer for receiving messages
    recv_buffers = [np.empty_like(chunk) for chunk in chunks]

    # the processes to communicate with
    send_to = (rank + 1) % num_processes
    recv_from = (rank - 1) % num_processes


    # n-1 iterations for operation
    for i in range(num_processes - 1):
        # prepare data to send
        send_idx = (rank - i) % num_processes
        msg = chunks[send_idx]
        req = comm.Isend(msg, send_to)

        # receive the data
        recv_idx = (rank - 1 - i) % num_processes # the index of the chunk to receive message and work with
        recv_req = comm.Irecv(recv_buffers[recv_idx], recv_from)
        recv_req.wait()

        # preform operation
        if recv_buffers[recv_idx].size != 0:
            #chunks[recv_idx] = op(chunks[recv_idx], recv_buffers[recv_idx])
            chunks[recv_idx] = np.vectorize(op)(chunks[recv_idx], recv_buffers[recv_idx])

    # n-1 iterations for distribution
    for i in range(num_processes - 1):
        # prepare data to send
        send_idx = (rank - i + 1) % num_processes
        msg = chunks[send_idx]
        req = comm.Isend(msg, send_to)

        # receive the data
        recv_idx = (rank - i) % num_processes  # the index of the chunk to receive message and work with
        recv_req = comm.Irecv(recv_buffers[recv_idx], recv_from)
        recv_req.wait()

        # save result
        if recv_buffers[recv_idx].size != 0:
            chunks[recv_idx] = recv_buffers[recv_idx]

    # put result into recv buffer
    recv[:] = np.concatenate(chunks, axis=0)
    return recv


def sum(x,y):
    return x+y

def mul(x,y):
    return x*y

def max(x,y):
    if x > y:
        return x
    else:
        return y

def min(x, y):
    if x < y:
        return x
    else:
        return y


def do_ringallreduce():
    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    send_data = np.array([[1,2,3,4,5], [1+rank,2+rank,3+rank,4+rank,5+rank], [10, -1, 2, rank, rank]])

    #sum
    recv_data = np.empty_like(send_data)
    ringallreduce(send_data, recv_data, comm, sum)

    print('====SUM==== \nrank: {} comm size: {}\n\n MY IMPL.\n send_data = {}\n recv_data = {}\n'.format(rank, size, send_data, recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.SUM)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    #mul
    recv_data = np.empty_like(send_data)
    ringallreduce(send_data, recv_data, comm, mul)
    print('====MUL==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank,size, send_data, recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.PROD)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    #max- works only elementwize, not on np array
    recv_data = np.empty_like(send_data)
    ringallreduce(send_data, recv_data, comm, max)
    print('====MAX==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank,size, send_data, recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.MAX)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    #min- works only elementwize, not on np array
    recv_data = np.empty_like(send_data)
    ringallreduce(send_data, recv_data, comm, MPI.MIN)
    print('====MIN==== \nrank: {} comm size: {}\n send_data = {}\n recv_data = {}\n'.format(rank,size, send_data, recv_data))
    recv_data = np.empty_like(send_data)
    comm.Allreduce(send_data, recv_data, op=MPI.MIN)
    print('MPI\n send_data = {}\n recv_data = {}\n\n'.format(send_data, recv_data))

    MPI.Finalize()


if __name__ == '__main__':
    start = time()
    do_ringallreduce()
    stop = time()
    print(f'runtime: {stop-start}[s]')

