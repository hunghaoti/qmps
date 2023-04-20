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

data_path = '../../data/20230420/data/'

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

def grad_descent(params, A, W, learn_rate, max_iter, tol = 1.0e-4): #1.-e-6 for 2pi case
    def get_grad(params):
        eps = 1.0e-3
        para_num = len(params)
        del_x = np.zeros(para_num)
        for i in range(para_num):
            p_m = list(params)
            p_p = list(params)
            p_m[i] = p_m[i] - eps
            p_p[i] = p_p[i] + eps
            f_minus = cost_func(p_m, A, W)
            f_plus = cost_func(p_p, A, W)
            del_x[i] = ((1.0/(2.0*eps)) * (f_plus - f_minus))
        #print(del_x)
        return del_x

    def get_grad_2pi(params):
        para_num = len(params)
        del_x = np.zeros(para_num)
        for i in range(para_num):
            p_m = list(params)
            p_p = list(params)
            p_m[i] = p_m[i] - 0.5 * np.pi
            p_p[i] = p_p[i] + 0.5 * np.pi
            f_minus = cost_func(p_m, A, W)
            f_plus = cost_func(p_p, A, W)
            del_x[i] = (0.5 * (f_plus - f_minus))
        #print(del_x)
        return del_x

    start = params
    x = start

    for _ in range(max_iter):
        diff = learn_rate * get_grad(x)
        crit = np.linalg.norm(diff)
        print('crit:', crit)
        if crit < tol:
            break
        old_cost = cost_func(x, A, W)
        x = x - diff
        cost = cost_func(x, A, W)
        print('step:', learn_rate)
        if cost > old_cost:
            x = x + diff
            learn_rate = learn_rate * 0.5
        print('iter:', _)
        print(cost_func(x, A, W))
    print("====")
    return x
    



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
np.random.seed(3)
params =[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.] #np.random.randn(N)
#res = minimize(obj_g, params, H0, options={'disp':True})
#params = res.x
A = iMPS([unitary_to_tensor(cirq.unitary(gate(params)))]).left_canonicalise()
print('energy:', A.energy([H0.to_matrix()]))
ps = [15]
f_e = open(data_path + "es.txt", "w")
for N in tqdm(ps):

    T = np.linspace(0, 10, 100)
    dt = T[1]-T[0]
    U = gate(params)

    WW = expm(-1j*H1.to_matrix()*2*dt)
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
        #U = param_unitary(params);
        U = gate(params)
        A_ = iMPS([unitary_to_tensor(cirq.unitary(U))]).left_canonicalise()
        en_old = en
        en = A_.energy([H0.to_matrix()])
        print(' energy', en)
        f_e.write(str(en) + '\n')
        f_e.flush()
        evs.append(A_.Es(ops))
        les.append(A_.overlap(A))
        resA = minimize(Aop_cost_func, par_Aop, (A_[0], H0, dt), 
                options={'disp':True})
        par_Aop = resA.x
        Aop = Hermit_gate(par_Aop)
        e_iA = expm(-1j*Aop*2*dt)
        res = minimize(cost_func, params, (A_[0], e_iA), 
                method = 'COBYLA', options={'disp':True})
        params = res.x
        #params = grad_descent(params, A_[0], WW, 0.1, 10000)
        #errs.append(res.fun)
        ps.append(params)
    f_e.close()

