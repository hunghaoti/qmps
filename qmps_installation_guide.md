# Installation guide 
- The following flow to guide you how to use the code from https://github.com/fergusbarratt/qmps
- We assume that you use the linux OS and 'python3' and 'pip' have been installed.
____________________________

## clone the repository
 Use git to clone the following repository in the same directory:
 1. https://github.com/fergusbarratt/qmps.git
 2. https://github.com/fergusbarratt/xmps.git
 
 You can use the following command:
```sh
git clone https://github.com/fergusbarratt/qmps.git
git clone https://github.com/fergusbarratt/xmps.git
```

## install the dependent module
The following modules need to be installed, including ['jaxlib'](https://pypi.org/project/jaxlib/), ['jax'](https://github.com/google/jax/), ['cirq'](https://quantumai.google/cirq). You can use the following command:
```sh
pip install jaxlib
pip install jax
pip install cirq
```

## install ['qmps'](https://github.com/fergusbarratt/qmps) and ['xmps'](https://github.com/fergusbarratt/xmps)
go to the corresponding directory and use pip to install them.

```sh
cd <DIR-YOU-CLONE-QMPS>/qmps
pip install -e .
cd <DIR-YOU-CLONE-QMPS>/xmps
pip install -e .
```
## Test to run the program
If you have done the above progress and you can try to run the code at :
`<DIR-YOU-CLONE-QMPS>`/qmps/new_tdvp/LoschmidtEchos.py
```sh
cd <DIR-YOU-CLONE-QMPS>/qmps/new_tdvp
python3 LoschmidtEchos.py
```

___________________________________ 
| Author | Date |
| ------ | ------ |
| Hao-Ti Hung | 2022/09/22 |


