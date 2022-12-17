from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.quantum_info.operators import Operator
import numpy as np

from xmps.iMPS import iMPS, Map 
from xmps.spin import U4
from xmps.iOptimize import find_ground_state
from xmps.spin import paulis

import cirq
from qmps.tools import unitary_to_tensor, environment_from_unitary, tensor_to_unitary, get_env_exact, get_env_exact_alternative
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from qmps.ground_state import Hamiltonian
from qmps.qiskit_qmps import time_evolve_cost_fun, param_unitary, time_evolve_measure_cost_fun
from qmps.represent import ShallowFullStateTensor

from tqdm import tqdm

from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.linalg import expm




def gate(v, symbol='U'):
    #return ShallowCNOTStateTensor(2, v)
    #return ShallowQAOAStateTensor(2, v)
    return ShallowFullStateTensor(2, v, symbol)
    #return FullStateTensor(U4(v))

def obj_g(p_, H):
    p, rs = p_[:15], p_[15:]
    A = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()
    Ham = [H.to_matrix()]
    return (A.energy(Ham))

## main function
g0, g1 = 1.0, 0.2
D = 2 #virtual bond dimension
N=15
H0 = Hamiltonian({'ZZ':-1.0, 'X':g0})
H1 = Hamiltonian({'ZZ':-1.0, 'X':g1})
params = np.random.randn(N)
res = minimize(obj_g, params, H0, options={'disp':True})
params = res.x
A = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()
print('ground state energy:', A.energy([H0.to_matrix()]))
lles = []
eevs = []
eers = []
ps = [15]
for N in tqdm(ps):

    T = np.linspace(0, 6, 300)
    dt = T[1]-T[0]
    U = gate(params)

    WW = expm(-1j*H1.to_matrix()*2*dt)
    ps = [params]
    ops = paulis(0.5)
    evs = []
    les = []
    errs = [res.fun]

    for _ in tqdm(T):
        #U = param_unitary(params);
        U = gate(params)
        A_ = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()
        evs.append(A_.Es(ops))
        les.append(A_.overlap(A))
        res = minimize(time_evolve_measure_cost_fun, params, (A_[0], WW), options={'disp':True})

        params = res.x
        errs.append(res.fun)
        ps.append(params)
    lles.append(les)
    eevs.append(evs)
    eers.append(errs)

ps = [15]
for q, i in enumerate(ps):
    j = int((np.max(list(ps))-i)/2)
    np.save('qiskit_lles', -np.log(np.array(lles)).T[0][:, j])

