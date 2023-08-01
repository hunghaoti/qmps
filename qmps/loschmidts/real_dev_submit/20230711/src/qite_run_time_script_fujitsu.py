from qiskit_ibm_runtime import Sampler, Options
from qiskit_qulacs import QulacsProvider
from mpi4py import MPI

import qite_qiskit

# Get rank in MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


backend = QulacsProvider().get_backend()  # Get QulacsBackend instance
options = Options(optimization_level=2)
sampler = Sampler(options=options, backend=backend)
qite_qiskit.main(sampler)

