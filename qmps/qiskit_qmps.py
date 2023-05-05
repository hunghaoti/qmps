from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.quantum_info.operators import Operator
import numpy as np
import cirq
from qmps.represent import FullStateTensor, Environment

from xmps.iMPS import iMPS, Map 
from xmps.spin import U4
from xmps.spin import paulis

from qmps.tools import unitary_to_tensor, environment_from_unitary, tensor_to_unitary, get_env_exact, get_env_exact_alternative
from qmps.time_evolve_tools import merge, put_env_on_left_site, put_env_on_right_site
from qmps.represent import ShallowFullStateTensor
from qiskit.primitives import Estimator
import os

def gate(v, symbol='U'):
    #return ShallowCNOTStateTensor(2, v)
    #return ShallowQAOAStateTensor(2, v)
    return ShallowFullStateTensor(2, v, symbol)
    #return FullStateTensor(U4(v))


def gate_to_operator(gate):
    """
    Returns a Qiskit Operator Object given a cirq Gate object
    """
    return Operator(cirq.unitary(gate))

def param_unitary(params, gate, inverse = False):
    """
    Build a parameterized unitary Operator from a set of parameters with a parameterization determined by gate.
    """
    return gate_to_operator(gate(params))

def partial_trace(ρ, n_spins, spins_keep):
    '''
    Perform a partial trace on a density matrix ρ. The size of ρ is 2^(n_spins) x 2^(n_spins), and the size of the space that is not traced out is 2^(len(spins_keep)) x 2^(len(spins_keep)). 
    
    This is needed because qiskit has no inbuilt ability to specify the reduced density matrices of a subset of the simulated qubits.
    '''
    reshaped = ρ.reshape(2*n_spins*[2])
    ket_index = [i for i in reversed(range(n_spins))]
    bra_index = [i + n_spins if i in spins_keep else i for i in reversed(range(n_spins))]
    
    einsum_index = ket_index + bra_index

    reduced = np.einsum(reshaped, einsum_index)

    output_dim = 2*len(spins_keep)
    output = reduced.reshape(output_dim, output_dim)
    return output

def represent_full_cost_function(params, U, gate):
    '''
    Cost function for variationally finding the environment of an iMPS with a quantum circuit. Requires full tomography of the reduced density matrices that are being evaluated.
    '''
    simulator = Aer.get_backend('statevector_simulator')
    target_environment = Operator(gate(params))
    
    circ_ρσ = QuantumCircuit(5)
    
    circ_ρσ.append(target_environment, [2,1])
    circ_ρσ.append(U, [1,0])
    circ_ρσ.append(target_environment, [4,3])
    
    result = execute(circ_ρσ, simulator).result()
    statevector = result.get_statevector(circ_ρσ)
    ρ = np.outer(statevector, statevector.conj())
    
    ρ_1 = partial_trace(ρ, 5, [0])
    ρ_2 = partial_trace(ρ, 5, [3])
    
    return np.linalg.norm(ρ_1-ρ_2)

def represent_sampled_circuit(params, U, shots, gate):
    '''
    Cost function for variationally finding the environment of an iMPS with a quantum circuit, without requiring the full reduced density matrix to be specified. Trace distance = Tr[|(ρ-σ)^2|] is used instead. 
    
    All estimated reduced density matrices are returned for the sake of debugging, see represent_sampled_cost_function() for the funciton to be used by optimizers. 
    '''
    simulator = Aer.get_backend('qasm_simulator')
    target_environment = Operator(gate(params))

    circ_ρσ = QuantumCircuit(5, 2)
    circ_ρρ = QuantumCircuit(4, 2)
    circ_σσ = QuantumCircuit(6, 2)
    
    circ_ρσ.append(target_environment, [2,1])
    circ_ρσ.append(U, [1,0])
    circ_ρσ.append(target_environment, [4,3])
    circ_ρσ.cx(0,3)
    circ_ρσ.h(0)
    circ_ρσ.measure([0, 3], [0,1])
    
    circ_ρρ.append(target_environment, [1,0])
    circ_ρρ.append(target_environment, [3,2])
    circ_ρρ.cx(0,2)
    circ_ρρ.h(0)
    circ_ρρ.measure([0,2], [0,1])

    circ_σσ.append(target_environment, [2,1])
    circ_σσ.append(U, [1,0])
    circ_σσ.append(target_environment, [5,4])
    circ_σσ.append(U, [4,3])
    circ_σσ.cx(0,3)
    circ_σσ.h(0)
    circ_σσ.measure([0,3], [0,1])

    result_ρσ = execute(circ_ρσ, simulator, shots=shots).result()
    counts_ρσ = result_ρσ.get_counts(circ_ρσ)
    
    result_ρρ = execute(circ_ρρ, simulator, shots=shots).result()
    counts_ρρ = result_ρρ.get_counts(circ_ρρ)
    
    result_σσ = execute(circ_σσ, simulator, shots=shots).result()
    counts_σσ = result_σσ.get_counts(circ_σσ)
    
    # If no 11s are measured then the test has failed 0 times so (1 - Tr[ρσ]) / 2 = 0 i.e. Tr[ρσ] = 1
    
    try:
        _ = counts_ρσ['11']
    except:
        counts_ρσ['11'] = 0
    
    try:
        _ = counts_σσ['11']
    except:
        counts_σσ['11'] = 0
    
    try:
        _ = counts_ρρ['11']
    except:
        counts_ρρ['11'] = 0
    
    ρρ = 1 - 2*(counts_ρρ['11']/shots)  # Trace [ρ^2]
    σσ = 1 - 2*(counts_σσ['11']/shots)  # Trace [σ^2]
    ρσ = 1 - 2*(counts_ρσ['11']/shots)  # Trace [ρσ]
    
    score = ρρ + σσ - 2 * ρσ  # Trace [|(ρ - σ)|^2]

    return [score, ρσ, ρρ, σσ]

def represent_sampled_cost_function(params, U):
    """
    Cost function that uses the circuit generated by represent_sampled_circuit() as a target to minimize. 
    """
    scores = represent_cost_function_full_return(params, U)
    return scores[0]

def simulate_state(U, V):
    """
    Function to return the state of the qmps in canonical form once of a given state and environment. Only works for 2 site Hamiltonians.
    """
    simulator = Aer.get_backend('statevector_simulator')
    
    c = QuantumCircuit(4)
    
    c.append(V, [3,2])
    c.append(U, [2,1])
    c.append(U, [1,0])
    
    result = execute(c, simulator).result()
    ψ = result.get_statevector(c)
    return ψ

def energy_cost_fun(params, H, gate):
    """
    energy minimized when looking for the ground state of a 2-site hamiltonian H.
    """
    U = gate(params)
    try:
        V = get_env_exact(U)
    except:
        V = get_env_exact_alternative(U)
    
    ψ = simulate_state(Operator(U), Operator(V))
    
    H = np.kron(np.kron(np.eye(2), H), np.eye(2))
    E = ψ.conj().T @ H @ ψ
    return E.real

def time_evolve_sim_state(params, A, WW):
    """
    Circuit which returns the overlap between 2 MPS states, defined by the unitaries U_ = U4(params) & U = tensor_to_unitary(A). The full wavefunction is returns for debugging purposes.
    """
    
    simulator = Aer.get_backend('statevector_simulator')
    circ = QuantumCircuit(6,6)
    
    U_ = gate(params)
    A_ = iMPS([unitary_to_tensor(cirq.unitary(U_))]).left_canonicalise()[0]
    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(A_, A_))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    U_ = Operator(tensor_to_unitary(A_))
    W = Operator(WW)
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))
    target_u = cirq.inverse(U_)

    circ.h(3)
    circ.cx(3,4)
    circ.unitary(U, [3,2])
    circ.unitary(U, [2,1])
    circ.unitary(W, [3,2])
    circ.unitary(L, [1,0])
    circ.unitary(R, [5,4])
    circ.unitary(target_u, [2,1])
    circ.unitary(target_u, [3,2])
    circ.cx(3,4)
    circ.h(3)
    # print(circ)
    
    result = execute(circ, simulator).result()
    ψ = result.get_statevector(circ)
    #print(ψ)
    
    return ψ

def time_evolve_sim_sample(params, A, WW, sampler_give=None):
    """
    Circuit which returns the overlap between 2 MPS states, defined by the unitaries U_ = U4(params) & U = tensor_to_unitary(A). The full wavefunction is returns for debugging purposes.
    """
    
    simulator = Aer.get_backend('statevector_simulator')
    circ = QuantumCircuit(6)
    
    U_ = gate(params)
    A_ = iMPS([unitary_to_tensor(cirq.unitary(U_))]).left_canonicalise()[0]
    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(A_, A_))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    U_ = Operator(tensor_to_unitary(A_))
    W = Operator(WW)
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))
    target_u = cirq.inverse(U_)

    circ.h(3)
    circ.cx(3,4)
    circ.unitary(U, [3,2])
    circ.unitary(U, [2,1])
    circ.unitary(W, [3,2])
    circ.unitary(L, [1,0])
    circ.unitary(R, [5,4])
    circ.unitary(target_u, [2,1])
    circ.unitary(target_u, [3,2])
    circ.cx(3,4)
    circ.h(3)
    result = execute(circ, simulator).result()
    ψ = result.get_statevector(circ)
    
    return ψ[0]

def time_evolve_cost_fun(params, A, W, sampler_give = None):
    """
    objective funciton that takes the probabiltiy of the all-zeros state from the wavefunction returns by time_evolve_sim_state(). This value is multiplied by 2 for normalization purposes.
    """
    cost = -np.sqrt(2*np.abs(time_evolve_sim_sample(params, A, W, sampler_give)))
    # print(cost)
    return cost

def time_evolve_sim_measure_sample(params, A, WW, sampler_give = None):
    """
    Circuit which returns the overlap between 2 MPS states, defined by the unitaries U_ = U4(params) & U = tensor_to_unitary(A). The full wavefunction is returns for debugging purposes.
    """
    

    circ = QuantumCircuit(6)
    
    U_ = gate(params)
    A_ = iMPS([unitary_to_tensor(cirq.unitary(U_))]).left_canonicalise()[0]
    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(A_, A_))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    U_ = Operator(tensor_to_unitary(A_))
    W = Operator(WW)
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))
    target_u = cirq.inverse(U_)

    circ.h(3)
    circ.cx(3,4)
    circ.unitary(U, [3,2])
    circ.unitary(U, [2,1])
    circ.unitary(W, [3,2])
    circ.unitary(L, [1,0])
    circ.unitary(R, [5,4])
    circ.unitary(target_u, [2,1])
    circ.unitary(target_u, [3,2])
    circ.cx(3,4)
    circ.h(3)

    circ.measure_all()
    shot_num = 8192

    # measurement
    if not sampler_give:
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(circ, simulator, shots = shot_num).result()
        counts = result.get_counts(circ)['000000']
        rate = np.sqrt(float(counts) / shot_num) # get amplitude
        #ψ = result.get_statevector(circ)
    else:
        sampler = sampler_give
        job = sampler.run(circ, shots = shot_num)
        print(f">>> Job ID: {job.job_id()}")
        print(f">>> Job Status: {job.status()}")
        result = job.result()
        rate = np.sqrt(result.quasi_dists[0][0]);
    
    return rate

def time_evolve_measure_cost_fun(params, A, W, sampler_give = None):
    """
    objective funciton that takes the probabiltiy of the all-zeros state from the wavefunction returns by measurement methods.
    """
    prob = time_evolve_sim_measure_sample(params, A, W, sampler_give)
    cost = -np.sqrt(2*prob)
    return cost

def time_evolve_sim_DN_sample(params, A, WW, N):
    # N is op number
    simulator = Aer.get_backend('statevector_simulator')
    circ = QuantumCircuit(2 * N + 4)
    
    U_ = gate(params)
    A_ = iMPS([unitary_to_tensor(cirq.unitary(U_))]).left_canonicalise()[0]
    E = Map(np.tensordot(WW, merge(A, A), [1, 0]), merge(A_, A_))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    U_ = Operator(tensor_to_unitary(A_))
    W = Operator(WW)
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))
    target_u = cirq.inverse(U_)

    idx_n = 1 + 2 * N
    circ.h(idx_n)
    circ.cx(idx_n, idx_n + 1)

    for i in range(idx_n, 1, -1):
        circ.unitary(U, [i, i-1])

    for i in range(1, N+1):
        circ.unitary(W, [i*2 + 1, i*2])

    circ.unitary(L, [1,0])
    circ.unitary(R, [idx_n + 2, idx_n + 1])

    for i in range(1, idx_n):
        circ.unitary(target_u, [i + 1, i])

    circ.cx(idx_n, idx_n + 1)
    circ.h(idx_n)

    result = execute(circ, simulator).result()
    ψ = result.get_statevector(circ)
    
    return ψ[0]

def time_evolve_DN_cost_fun(params, A, W, N):
    cost = -np.sqrt(2*np.abs(time_evolve_sim_DN_sample(params, A, W, N)))
    # print(cost)
    return cost
