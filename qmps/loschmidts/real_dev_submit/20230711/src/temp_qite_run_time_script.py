import sys
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Estimator, Options

#sys.path.insert(0, "..")  # Add source_program directory to the path

import qite_qiskit
from qiskit import Aer
from qiskit_ibm_runtime import RuntimeEncoder, RuntimeDecoder
from qiskit_ibm_runtime.program import UserMessenger


inputs = {"iterations": 3}
options = Options()
options.optimization_level = 2


service = QiskitRuntimeService()
with Session(service=service, backend="%device_name") as session:
    sampler = Sampler(session=session, options=options)
    qite_qiskit.main(sampler)
    session.close()
