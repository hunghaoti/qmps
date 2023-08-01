import configparser
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options

import qite_qiskit

config = configparser.ConfigParser()
config.read(Path(__file__).parent / 'config.ini')

options = Options(optimization_level=2)
service = QiskitRuntimeService(channel='ibm_quantum',
                               token=config['secret']['ibm_token'])
with Session(service=service, backend=config['device']['name']) as session:
    sampler = Sampler(session=session, options=options)
    qite_qiskit.main(sampler)
    session.close()
