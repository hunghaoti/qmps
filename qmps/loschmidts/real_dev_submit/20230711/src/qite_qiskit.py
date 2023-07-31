from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator, UnitarySimulator
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


#cost_func = time_evolve_cost_fun
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
    A_op[0, 1] = par[4] + 1j*par[10]
    A_op[0, 2] = par[5] + 1j*par[11]
    A_op[0, 3] = par[6] + 1j*par[12]
    A_op[1, 2] = par[7] + 1j*par[13]
    A_op[1, 3] = par[8] + 1j*par[14]
    A_op[2, 3] = par[9] + 1j*par[15]
    #lower left
    A_op[1, 0] = par[4] - 1j*par[10]
    A_op[2, 0] = par[5] - 1j*par[11]
    A_op[3, 0] = par[6] - 1j*par[12]
    A_op[2, 1] = par[7] - 1j*par[13]
    A_op[3, 1] = par[8] - 1j*par[14]
    A_op[3, 2] = par[9] - 1j*par[15]
    return A_op

def CX_tens():
    cx_tens = np.zeros((4, 4))
    cx_tens[0, 0] = 1.0
    cx_tens[1, 3] = 1.0
    cx_tens[2, 2] = 1.0
    cx_tens[3, 1] = 1.0
    cx_tens = cx_tens.reshape(2, 2, 2, 2)
    return cx_tens

def h_tens(): # hadamard
    h_tens = np.zeros((2, 2))
    h_tens[0, 0] = 1.0
    h_tens[0, 1] = 1.0
    h_tens[1, 0] = 1.0
    h_tens[1, 1] = -1.0
    h_tens = h_tens * pow(2, -0.5)
    return h_tens

def i_ten():
    n = 6
    dim = pow(2, 6)
    i_tens = np.zeros((dim, dim))
    for i in range(0, dim):
        i_tens[i, i] = 1.0
    i_tens = i_tens.reshape(2, 2, 2, 2, 2, 2,   2, 2, 2, 2, 2, 2)
    i_tens = i_tens.astype(complex)
    return i_tens;


def Aop_sample(par_Aop, A, H, dt):
    Aop = Hermit_gate(par_Aop)
    A1 = expm(-1*H.to_matrix()*2*dt)
    A2 = np.zeros((4, 4))
    A2[0,0]=1.0
    A2[1,1]=1.0
    A2[2,2]=1.0
    A2[3,3]=1.0

    eiA = expm(1j*Aop*2*dt)
    e_iA = expm(-1j*Aop*2*dt)
    e_H = expm(-1*H.to_matrix()*2*dt)
    Op = - eiA @ e_H - e_H @ e_iA


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
    tens_cx = CX_tens()
    tens_h = h_tens()

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
    val = 1.0 * tmp[0,0,0,0,0,0]
    #print('val',val)


    #np.einsum()
    return val.real

def Aop_cost_func(par_Aop, A, H, dt):
    prob = Aop_sample(par_Aop, A, H, dt)
    cost = prob
    return cost

def TN_Measure(par, par_prime, Op):
    U  = gate(par)
    Up = gate(par_prime)
    A  = iMPS([unitary_to_tensor(cirq.unitary(U ))]).left_canonicalise()[0]
    Ap = iMPS([unitary_to_tensor(cirq.unitary(Up))]).left_canonicalise()[0]
    E = Map(np.tensordot(Op, merge(A, A), [1, 0]), merge(Ap, Ap))
    x, r = E.right_fixed_point()
    U  = Operator(tensor_to_unitary(A ))
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
    tens_cx = CX_tens()
    tens_h = h_tens()

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
    val = 1.0 * tmp[0,0,0,0,0,0]
    #print('val',val)


    #np.einsum()
    return val.real

def MPS_Measure(par, par_prime, Op):
    U  = gate(par)
    Up = gate(par_prime)
    A  = iMPS([unitary_to_tensor(cirq.unitary(U ))]).left_canonicalise()[0]
    Ap = iMPS([unitary_to_tensor(cirq.unitary(Up))]).left_canonicalise()[0]
    E = Map(A, Ap)
    xr, r = E.right_fixed_point()
    xl, l = E.left_fixed_point()
    d, D = 2, 2
    op = Op.reshape(d, d, d, d)
    lAAHAAr = ncon([l] + [A, A] + [op] + [Ap.conj(), + Ap.conj()] + [r], 
            [[1,2],[3,1,7],[5,7,9],[3,5,4,6],[4,2,8],[6,8,10],[9,10]])
    lr = ncon([l,r], [[1,2],[1,2]])
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
def main(sampler=None):
    base_folder = Path(__file__).parent

    config_path = base_folder / 'config.ini'
    config = ConfigParser()
    config.read(config_path)

    data_base_folder = base_folder / config['path']['data_path']
    previous_data_index = int(config["runtime_state"]["data_index"])
    data_input_folder = data_base_folder / f'data{previous_data_index}'
    data_output_folder = data_base_folder / f'data{previous_data_index + 1}'
    config['runtime_state']['data_index'] = str(previous_data_index + 1)
    with open(config_path, 'w') as configfile:
        config.write(configfile)

    os.makedirs(data_output_folder / 'params', exist_ok=True)
    os.makedirs(data_output_folder / 'A_params', exist_ok=True)


    g0, g1 = 1.0, 0.2
    D = 2 #virtual bond dimension
    N=15
    H0 = Hamiltonian({'ZZ':-1.0, 'X':g0})
    #H1 = Hamiltonian({'ZZ':-1.0, 'X':g1})
    np.random.seed(3)
    dt=float(config['parameter']['dt'])
    cnt=int(config['runtime_state']['loop_index'])
    if cnt == 0: # the first run
        params =[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    else:
        params = np.loadtxt(data_input_folder / 'params' / f'params_tau{round(cnt*dt, 2)}')

    #np.random.randn(N)
    termi_t = float(config['parameter']['terminate_time'])
    tmp = unitary_to_tensor(cirq.unitary(gate(params)))
    A = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()
    print('energy:', A.energy([H0.to_matrix()]))
    ps = [15]


    for N in tqdm(ps):
    
        T = np.linspace(0, termi_t - dt, int(termi_t/dt - 0.0001))
        dt = T[1]-T[0]
        print('dt', dt)
        U = gate(params)
    
        #WW = expm(-1j*H1.to_matrix()*2*dt)
        ps = [params]
        ops = paulis(0.5)
        evs = []
        les = []
        np.random.seed(0)
        par_Aop = np.random.randn(16)
    
        #errs = [res.fun]
        en_old = 100.0
        en = 100.0
    
        for _ in tqdm(T):
            #save current stage
            with open(config_path, "w") as configfile:
                config.write(configfile)

            #U = param_unitary(params);
            t=dt*cnt
            t_str=str(round(t, 2))
            with open(data_output_folder / "params" / f"params_tau{t_str}", "w") as f_pars:
                np.savetxt(f_pars, params)
            U = gate(params)
            A_ = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()
            en_old = en
            H0_mat = H0.to_matrix()
            HH_mat = H0_mat @ H0_mat
            en = A_.energy([H0_mat])
            ee = A_.energy([HH_mat])
            # s = A_.entropy()
            var = ee - en*en
            print('var', var)
            
            print(' energy', en)
            # print(' entropy', s)

            with (open(data_output_folder / "es.txt", "a") as f_e,
                  open(data_output_folder / "s.txt", "a") as f_s,
                  open(data_output_folder / "var.txt", "a") as f_var
                ):
                f_e.write(f'{en}\n')
                # f_s.write(f'{s}\n')
                f_var.write(f'{var}\n')

            evs.append(A_.Es(ops))
            les.append(A_.overlap(A))
            resA = minimize(Aop_cost_func, par_Aop, (A_[0], H0, dt), 
                    options={'disp':True})
            par_Aop = resA.x
            with  open(data_output_folder / 'A_params' / f'params_tau{t_str}', 'w') as f_A_pars:
                np.savetxt(f_A_pars, par_Aop)
            
            #A error 
            e_H = expm(-1*H0_mat*4.0*dt)
            Norm = TN_Measure(params, params, e_H)
            A_err = 2.0 +  (1.0 / np.sqrt(Norm)) * Aop_cost_func(par_Aop, A_[0], H0, dt)
           
            with open(data_output_folder / "A_err.txt", "w") as f_Aerr:
                f_Aerr.write(f'{A_err}\n')

            #A operate on circuit
            Aop = Hermit_gate(par_Aop)
            e_iA = expm(-1j*Aop*2*dt)
            res = minimize(cost_func, params, (A_[0], e_iA, sampler), 
                    method = 'COBYLA', options={'disp':True, 'tol':1.0e-3, 'catol':1.0e-3})
            params = res.x
            
            #params = grad_descent(params, A_[0], WW, 0.1, 10000)
            #errs.append(res.fun)
            ps.append(params)
            cnt = cnt + 1
            config['runtime_state']['loop_index'] = str(cnt)
    

if __name__ == "__main__":
    main()
