import configparser
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

config = configparser.ConfigParser()
config.read(Path(__file__).parent / 'config.ini')

# Save an IBM Quantum account.
QiskitRuntimeService.save_account(channel='ibm_quantum',
                                  token=config['secret']['ibm_token'])
