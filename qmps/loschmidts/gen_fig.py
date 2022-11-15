import numpy as np
import matplotlib.pyplot as plt

ls1 = np.load('qiskit_lles.npy')
ls2 = np.load('lles.npy')
dt = 0.02
length = len(ls1)
xs = np.arange(0, dt * length, dt)
plt.plot(xs, ls2, '-', linewidth = 3.0, color = 'orange', label = 'cirq')
plt.plot(xs, ls1, '--', linewidth = 1.5, color = 'tab:blue', label = 'qiskit')
plt.title('The Loschmidt echo of the Ising model on different platform.')
plt.legend(loc='upper right')
plt.xlabel('t')
plt.ylabel('Loschmidt echo')
plt.show()

