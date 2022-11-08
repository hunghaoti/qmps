import numpy as np
import matplotlib.pyplot as plt


ls = np.load('lles.npy')
dt = 0.02

length = len(ls)
xs = np.arange(0, dt * length, dt)
plt.plot(xs, ls)
plt.title('Trasverse field Ising model Loschmidt echo')
plt.xlabel('t')
plt.ylabel('Loschmidt echo')
plt.show()

