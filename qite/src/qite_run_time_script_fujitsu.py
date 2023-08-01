from mpi4py import MPI
from qiskit_qulacs import QulacsProvider
from qiskit.primitives import BackendSampler

import qite_qiskit

# Get rank in MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

backend = QulacsProvider().get_backend()  # Get QulacsBackend instance
sampler = BackendSampler(backend=backend)
qite_qiskit.main(sampler)
