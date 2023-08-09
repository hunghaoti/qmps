# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
# from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
from qiskit.quantum_info.operators import Operator
import numpy as np
import os
from configparser import ConfigParser
from pathlib import Path

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
from ncon import ncon
from numpy import trace as tr
import matplotlib.pyplot as plt

from klepto import no_cache
from klepto.archives import dir_archive
from klepto.keymaps import hashmap

# The default keymap uses built-in hash function which is seeded by a random
# number for each process.
stable_keymap = hashmap(flat=True, algorithm='md5')

# cost_func = time_evolve_cost_fun
cost_func = time_evolve_measure_cost_fun


def Hermit_gate(par):
    # par has a 10-elem
    A_op = np.zeros(16)
    A_op = A_op.astype(complex)
    A_op = A_op.reshape(4, 4)
    #diag term
    A_op[0, 0] = par[0]
    A_op[1, 1] = par[1]
    A_op[2, 2] = par[2]
    A_op[3, 3] = par[3]

    #off dig term
    #upper right
    A_op[0, 1] = par[4] + 1j * par[10]
    A_op[0, 2] = par[5] + 1j * par[11]
    A_op[0, 3] = par[6] + 1j * par[12]
    A_op[1, 2] = par[7] + 1j * par[13]
    A_op[1, 3] = par[8] + 1j * par[14]
    A_op[2, 3] = par[9] + 1j * par[15]
    #lower left
    A_op[1, 0] = par[4] - 1j * par[10]
    A_op[2, 0] = par[5] - 1j * par[11]
    A_op[3, 0] = par[6] - 1j * par[12]
    A_op[2, 1] = par[7] - 1j * par[13]
    A_op[3, 1] = par[8] - 1j * par[14]
    A_op[3, 2] = par[9] - 1j * par[15]
    return A_op


def cnot_tensor():
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]).reshape(2, 2, 2, 2)


def hadamard_tensor():
    return 2**-0.5 * np.array([
        [1.0, 1.0],
        [1.0, -1.0],
    ])


def i_ten():
    n = 6
    i_tens = np.identity(2**n, dtype=complex)
    i_tens = i_tens.reshape(*[2] * 2 * n)
    # i_tens = i_tens.astype(complex)
    return i_tens


def Aop_sample(par_Aop, A, H, dt):
    hermitianized = Hermit_gate(par_Aop)
    A1 = expm(-1 * H.to_matrix() * 2 * dt)
    A2 = np.identity(4, dtype=np.double)

    eiA = expm(1j * hermitianized * 2 * dt)
    e_iA = expm(-1j * hermitianized * 2 * dt)
    e_H = expm(-1 * H.to_matrix() * 2 * dt)
    Op = -eiA @ e_H - e_H @ e_iA

    E = Map(np.tensordot(Op, merge(A, A), [1, 0]), merge(A, A))
    Op = Op.reshape(2, 2, 2, 2)
    Op = np.transpose(Op, (1, 0, 3, 2))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))
    target_u = cirq.inverse(U)

    tens_U = U.data.reshape(2, 2, 2, 2)
    tens_U = np.transpose(tens_U, (1, 0, 3, 2))
    tens_R = R.data.reshape(2, 2, 2, 2)
    tens_R = np.transpose(tens_R, (1, 0, 3, 2))
    tens_L = L.data.reshape(2, 2, 2, 2)
    tens_L = np.transpose(tens_L, (1, 0, 3, 2))
    tens_Up = target_u.data.reshape(2, 2, 2, 2)
    tens_Up = np.transpose(tens_Up, (1, 0, 3, 2))
    tens_cx = cnot_tensor()
    tens_h = hadamard_tensor()

    tmp = i_ten()
    dim = pow(2, 6)
    vec_i = np.zeros(dim)
    vec_i[0] = 1.0
    vec_i = vec_i.reshape(2, 2, 2, 2, 2, 2)

    tmp = np.einsum('abcdefghijkl,cm->abmdefghijkl', tmp, tens_h)
    tmp = np.einsum('abcdefghijkl,bcmn->amndefghijkl', tmp, tens_cx)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, tens_Up)
    tmp = np.einsum('abcdefghijkl,demn->abcmnfghijkl', tmp, tens_Up)
    tmp = np.einsum('abcdefghijkl,abmn->mncdefghijkl', tmp, tens_R)
    tmp = np.einsum('abcdefghijkl,efmn->abcdmnghijkl', tmp, tens_L)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, Op)
    tmp = np.einsum('abcdefghijkl,demn->abcmnfghijkl', tmp, tens_U)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, tens_U)
    tmp = np.einsum('abcdefghijkl,bcmn->amndefghijkl', tmp, tens_cx)
    tmp = np.einsum('abcdefghijkl,cm->abmdefghijkl', tmp, tens_h)
    tmp = np.einsum('abcdef,abcdefghijkl->ghijkl', vec_i, tmp)
    #print('circ',tmp.reshape(8, 8))
    val = 1.0 * tmp[0, 0, 0, 0, 0, 0]
    #print('val',val)

    #np.einsum()
    return val.real


def Aop_cost_func(par_Aop, A, H, dt):
    prob = Aop_sample(par_Aop, A, H, dt)
    cost = prob
    return cost


def TN_Measure(par, par_prime, Op):
    U = gate(par)
    Up = gate(par_prime)
    A = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()[0]
    Ap = iMPS([unitary_to_tensor(cirq.unitary(Up))]).left_canonicalise()[0]
    E = Map(np.tensordot(Op, merge(A, A), [1, 0]), merge(Ap, Ap))
    x, r = E.right_fixed_point()
    U = Operator(tensor_to_unitary(A))
    Up = Operator(tensor_to_unitary(Ap))
    R = Operator(put_env_on_left_site(r))
    L = Operator(put_env_on_right_site(r.conj().T))

    Op = Op.reshape(2, 2, 2, 2)
    Op = np.transpose(Op, (1, 0, 3, 2))
    target_u = cirq.inverse(Up)

    tens_U = U.data.reshape(2, 2, 2, 2)
    tens_U = np.transpose(tens_U, (1, 0, 3, 2))
    tens_R = R.data.reshape(2, 2, 2, 2)
    tens_R = np.transpose(tens_R, (1, 0, 3, 2))
    tens_L = L.data.reshape(2, 2, 2, 2)
    tens_L = np.transpose(tens_L, (1, 0, 3, 2))
    tens_Up = target_u.data.reshape(2, 2, 2, 2)
    tens_Up = np.transpose(tens_Up, (1, 0, 3, 2))
    tens_cx = cnot_tensor()
    tens_h = hadamard_tensor()

    tmp = i_ten()
    dim = pow(2, 6)
    vec_i = np.zeros(dim)
    vec_i[0] = 1.0
    vec_i = vec_i.reshape(2, 2, 2, 2, 2, 2)

    tmp = np.einsum('abcdefghijkl,cm->abmdefghijkl', tmp, tens_h)
    tmp = np.einsum('abcdefghijkl,bcmn->amndefghijkl', tmp, tens_cx)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, tens_Up)
    tmp = np.einsum('abcdefghijkl,demn->abcmnfghijkl', tmp, tens_Up)
    tmp = np.einsum('abcdefghijkl,abmn->mncdefghijkl', tmp, tens_R)
    tmp = np.einsum('abcdefghijkl,efmn->abcdmnghijkl', tmp, tens_L)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, Op)
    tmp = np.einsum('abcdefghijkl,demn->abcmnfghijkl', tmp, tens_U)
    tmp = np.einsum('abcdefghijkl,cdmn->abmnefghijkl', tmp, tens_U)
    tmp = np.einsum('abcdefghijkl,bcmn->amndefghijkl', tmp, tens_cx)
    tmp = np.einsum('abcdefghijkl,cm->abmdefghijkl', tmp, tens_h)
    tmp = np.einsum('abcdef,abcdefghijkl->ghijkl', vec_i, tmp)
    #print('circ',tmp.reshape(8, 8))
    val = 1.0 * tmp[0, 0, 0, 0, 0, 0]
    #print('val',val)

    #np.einsum()
    return val.real


def MPS_Measure(par, par_prime, Op):
    U = gate(par)
    Up = gate(par_prime)
    A = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()[0]
    Ap = iMPS([unitary_to_tensor(cirq.unitary(Up))]).left_canonicalise()[0]
    E = Map(A, Ap)
    xr, r = E.right_fixed_point()
    xl, l = E.left_fixed_point()
    d, D = 2, 2
    op = Op.reshape(d, d, d, d)
    lAAHAAr = ncon([l] + [A, A] + [op] + [Ap.conj(), +Ap.conj()] + [r],
                   [[1, 2], [3, 1, 7], [5, 7, 9], [3, 5, 4, 6], [4, 2, 8],
                    [6, 8, 10], [9, 10]])
    lr = ncon([l, r], [[1, 2], [1, 2]])
    res = lAAHAAr / lr
    return res


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


def obj_s(p_):
    p = p_[:15]
    A = iMPS([unitary_to_tensor(cirq.unitary(gate(p)))]).left_canonicalise()
    A = A.left_canonicalise()
    return -(A.entropy())


## main function
def main(sampler=None, cache_folder_name=''):
    base_folder = Path(__file__).parent

    config_path = base_folder / 'config.ini'
    config = ConfigParser()
    config.read(config_path)

    data_base_folder = base_folder / config['path']['data_path']
    cache_folder = data_base_folder / cache_folder_name

    dt = float(config['parameter']['dt'])
    terminate_time = float(config['parameter']['terminate_time'])

    g0, g1 = 1.0, 0.2
    H0 = Hamiltonian({'ZZ': -1.0, 'X': g0})
    H0_mat = H0.to_matrix()
    HH_mat = H0_mat @ H0_mat
    pauli_xyz = paulis(0.5)
    evs = []
    les = []

    @no_cache(cache=dir_archive(name=cache_folder),
              keymap=stable_keymap,
              ignore=('params', 'A_elements'))
    def evolve(end_time: float,
               step: float,
               step_count: int,
               params: np.ndarray[np.float64] = None,
               A_elements: np.ndarray[np.float64] = None):
        '''
        Args:
            step_count: the number of steps passed
        '''
        if step_count == 0:
            return {
                'params': np.ones(15),
                'A_elements': np.random.randn(16),
            }
        if params is None or A_elements is None:
            raise ValueError('params and A_elements should not be None')

        U = gate(params)
        A_ = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()

        # Calculate energy, entropy and variance of energy
        energy = A_.energy([H0_mat])
        ee = A_.energy([HH_mat])
        # entropy = A_.entropy()
        variance = ee - energy * energy

        # Fit A operator
        # evs.append(A_.Es(pauli_xyz))
        # les.append(A_.overlap(A))  # get overlap with previous A?
        resA = minimize(Aop_cost_func,
                        A_elements, (A_[0], H0, step),
                        options={'disp': True})
        A_elements = resA.x

        # error of A operator
        e_H = expm(-1 * H0_mat * 4.0 * step)
        Norm = TN_Measure(params, params, e_H)
        A_err = 2.0 + (1.0 / np.sqrt(Norm)) * Aop_cost_func(
            A_elements, A_[0], H0, step)

        # A operates on circuit
        A_operator = Hermit_gate(A_elements)
        e_iA = expm(-1j * A_operator * 2 * step)
        res = minimize(cost_func,
                       params, (A_[0], e_iA, sampler),
                       method='COBYLA',
                       options={
                           'disp': True,
                           'tol': 1.0e-3,
                           'catol': 1.0e-3
                       })
        params = res.x

        #params = grad_descent(params, A_[0], WW, 0.1, 10000)
        #errs.append(res.fun)
        return {
            'energy': energy,
            'variance': variance,
            'params': params,
            'A_elements': A_elements,
            'A_error': A_err,
        }

    @no_cache(cache=dir_archive(name=cache_folder),
              keymap=stable_keymap,
              ignore=('params', 'A_elements'))
    def get_energies(
        end_time: float,
        step: float,
    ):
        bound_dimension = 2  #virtual bond dimension
        N = 15
        #H1 = Hamiltonian({'ZZ':-1.0, 'X':g1})
        A = iMPS([unitary_to_tensor(cirq.unitary(gate([1.] * 15)))
                 ]).left_canonicalise()

        # print('energy:', A.energy([H0.to_matrix()]))
        something = [15]

        for N in tqdm(something):
            #WW = expm(-1j*H1.to_matrix()*2*dt)

            # XXX: Why to seed the random number generator with a constant here?
            np.random.seed(0)

            params = None
            A_elements = None
            energies = []
            for step_count in tqdm(range(int(np.ceil(end_time / step)) + 1)):
                result = evolve(end_time,
                                step,
                                step_count,
                                params=params,
                                A_elements=A_elements)
                params = result['params']
                A_elements = result['A_elements']
                if 'energy' in result:
                    energies.append(result['energy'])

            xs = np.arange(0, step * len(energies), step) + step
            plt.plot(xs,
                     energies,
                     '-',
                     linewidth=3.0,
                     color='orange',
                     label='energy')
            plt.title('energy in evolution')
            plt.legend(loc='upper right')
            plt.xlabel('evolved time')
            plt.ylabel('energy')
            plt.savefig(cache_folder / 'energy.png')
            return [xs, energies]

    return get_energies(terminate_time, dt)


if __name__ == '__main__':
    main()