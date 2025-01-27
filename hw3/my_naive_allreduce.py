import numpy as np
import mpi4py

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
    all_values = [np.empty(recv.shape) for _ in range(size)]

    # send buffer to all other processes, and copy the send_buffer for the current process.
    send_requests = []
    for i in range(size):
        if i != rank:
            req = comm.Isend(send, dest=i)
            send_requests.append(req)

    # receive back shared data from other processes
    receive_requests = []
    for i in range(size):
        if i == rank:
            all_values[i] = send
        else:
            receive_requests.append(comm.Irecv(all_values[i], i))

    # wait for all the values!
    for i in range(size):
        if i != rank:
            receive_requests[i].wait()

    # reduce everyone
    compute = all_values[0]
    for i in range(1, size):
        compute = op(all_values[i], compute)

    recv[:] = compute
    return recv


def sum(x,y):
    return x+y

def mul(x,y):
    return x*y


def test():
    MPI.Init()

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    send_data = np.random.rand(100, 100, 100)

    # test sum
    recv_data = np.empty_like(send_data)
    recv_mpi = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, sum)
    comm.Allreduce(send_data, recv_mpi, op=MPI.SUM)
    if not np.allclose(recv_data, recv_mpi):
        print('failed sum')
        exit(0)

    #test mul
    recv_data = np.empty_like(send_data)
    recv_mpi = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, mul)
    comm.Allreduce(send_data, recv_mpi, op=MPI.PROD)
    if not np.allclose(recv_data, recv_mpi):
        print('failed mul')
        exit(0)

    #test max
    recv_data = np.empty_like(send_data)
    recv_mpi = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, max)
    comm.Allreduce(send_data, recv_mpi, op=MPI.MAX)
    if not np.allclose(recv_data, recv_mpi):
        print('failed max')
        exit(0)

    #test mul
    recv_data = np.empty_like(send_data)
    recv_mpi = np.empty_like(send_data)
    allreduce(send_data, recv_data, comm, min)
    comm.Allreduce(send_data, recv_mpi, op=MPI.MIN)
    if not np.allclose(recv_data, recv_mpi):
        print('failed min')
        exit(0)

    print('all passed!')

    MPI.Finalize()

if __name__ == '__main__':
    test()
