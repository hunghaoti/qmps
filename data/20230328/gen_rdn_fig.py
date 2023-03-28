import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def GetDataFromFile(file_str, every_n_line):
    file1 = open(file_str, 'r')
    Lines = file1.readlines()
    allList = []
    for subline in range(0, every_n_line):
        allList.append([])
    cnt = 0
    for line in Lines:
        v = float(line)
        list_idx = cnt % every_n_line
        allList[list_idx].append(v)
        cnt += 1
    return allList


data_path = './data/'
exact_energy = -1.2725424849552869

for rnd in range(0, 5):
    file_name = data_path + 'energies_rdn' + str(rnd) + '.txt'
    es = GetDataFromFile(file_name, 1)[0]
    dt = 0.1
    data_len = len(es)
    ts = np.arange(0, data_len * dt, dt)
    plt.plot(ts, es, label = 'rand' + str(rnd))
plt.axhline( y = exact_energy, color = 'black', linestyle = '--', label = 'exact')
plt.ylim([-1.4, 0.4])
plt.legend(loc = 'upper right')
plt.title('Start QITE from different random initial')
plt.xlabel('imaginary time $\\tau$')
plt.ylabel('energy')
plt.savefig('./figs/rnd_init.png')
plt.show()

plt.close()
file_name = data_path + 'energies_eye.txt'
es = GetDataFromFile(file_name, 1)[0]
dt = 0.1
data_len = len(es)
ts = np.arange(0, data_len * dt, dt)
plt.ylim([-1.4, 0.4])
plt.plot(ts, es)
plt.axhline( y = exact_energy, color = 'black', linestyle = '--', label = 'exact')
plt.legend(loc = 'upper right')
plt.title('Start QITE from params all one')
plt.xlabel('imaginary time $\\tau$')
plt.ylabel('energy')
plt.savefig('./figs/eye_init.png')
plt.show()

plt.close()
file_name = data_path + 'energies_gd.txt'
es = GetDataFromFile(file_name, 1)[0]
dt = 0.1
data_len = len(es)
ts = np.arange(0, data_len * dt, dt)
plt.ylim([-1.4, 0.4])
plt.plot(ts, es)
plt.axhline( y = exact_energy, color = 'black', linestyle = '--', label = 'exact')
plt.legend(loc = 'upper right')
plt.title('Start QITE from ground state')
plt.xlabel('imaginary time $\\tau$')
plt.ylabel('energy')
plt.savefig('./figs/gd_init.png')
plt.show()

plt.close()
file_name = data_path + 'energies_eye.txt'
es = GetDataFromFile(file_name, 1)[0]
dt = 0.1
data_len = len(es)
ts = np.arange(0, data_len * dt, dt)
plt.ylim([-1.4, 0.4])
plt.plot(ts, es, label = '$d\\tau=0.1$')
file_name = data_path + 'energies_eye_dt0.01.txt'
es2 = GetDataFromFile(file_name, 1)[0]
dt = 0.01
data_len = len(es2)
ts = np.arange(0, data_len * dt, dt)
plt.plot(ts, es2, label = '$d\\tau=0.01$')
plt.axhline( y = exact_energy, color = 'black', linestyle = '--', label = 'exact')
plt.legend(loc = 'upper right')
plt.title('Compare different $d\\tau$')
plt.xlabel('imaginary time $\\tau$')
plt.ylabel('energy')
plt.savefig('./figs/Cmp_diff_dt.png')
plt.show()
