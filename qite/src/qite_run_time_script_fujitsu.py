import matplotlib.pyplot as plt
from mpi4py import MPI
from qiskit_qulacs import QulacsProvider
from qiskit.primitives import BackendSampler

import qite_qiskit

# Get rank in MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpisize = comm.Get_size()

backend = QulacsProvider().get_backend()  # Get QulacsBackend instance
sampler = BackendSampler(backend=backend)
result = qite_qiskit.main(sampler, f'rank{rank:02}')

if rank == 0:
    all_results = [result]
    for i in range(1, mpisize):
        all_results.append(comm.recv(None, i))
    for xs, energies in all_results:
        plt.plot(xs, energies, '-')
    plt.title('energy in evolution')
    plt.legend(loc='upper right')
    plt.xlabel('evolved time')
    plt.ylabel('energy')
    plt.savefig('all_energy.png')
else:
    comm.send(result, 0)
