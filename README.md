# Representing, optimizing and evolving matrix product states on NISQ devices

## Installation

1. Clone this somewhere.
2. Install [Poetry](https://python-poetry.org/docs/#installation)
3. Run the commands below from this folder.

```
poetry install
pip install -e .
```

## Prepare

Copy `qite/src/config.example.ini` to `qite/src/config.ini` and put your IBM API token into `config.ini`. Get your IBM API token at the [IBM Quantum Account page](https://quantum-computing.ibm.com/account).

## Instructions for Running on Fujitsu's System

The architecture of Fujitsu's login server is x64, and the architecture of the Slurm system is ARM64, however, they share the same file system, so installing an executable program on the login server may break the environment of the Slurm system. Here are some tips to make the programs run on both sides well.

### pyenv

[pyenv](https://github.com/pyenv/pyenv) doesn't contain executable binary files, so you only need to install it on the login server to access it on both sides.

Because the login server and the Slurm system share the same file system, you have to install Python to the different folders. On the login server, you can install Python as usual, for example.

```
pyenv install 3.11.4
```

Before installing Python on the Slurm system, add the below script at the end of `~/.bashrc`, or `~/.zshrc` if you use zsh. This script will help you switch to the ARM version of Python when invoking the Slurm shell.

```
if [ -n "$SLURM_PROCID" ];then
    pyenv shell "$(pyenv version-name)-arm"
    export POETRY_CONFIG_DIR="$HOME/.config/pypoetry-arm"
    export POETRY_HOME="$HOME/.local/share/pypoetry-arm"
    export POETRY_CACHE_DIR="$HOME/.cache/pypoetry-arm"
fi
```

To install Python on the Slrum system use this command on the Slrum shell instead. Replace `{{PYTHON_VERSION`}}` with the version name you want to install.

```
$(pyenv root)/plugins/python-build/bin/python-build {{PYTHON_VERSION}} $(pyenv root)/versions/{{PYTHON_VERSION}}-arm
```

### Poetry

You must add the script to `~/.bashrc` or `~/.zshrc` before installing [Poetry](https://python-poetry.org/docs/#installation) on the Slurm system. Make sure you have switched the Python version you want by pyenv. The next step, use

```
curl -sSL https://install.python-poetry.org | python3 - --git https://github.com/python-poetry/poetry.git@master
```

on the login server to install Poetry on the login server and then use the same command on the Slurm shell to install Poetry on the Slurm system.

The reason that to install Poetry from git is the private PyPI server only provides legacy MD5 hashes, and due to [poetry/#6301](https://github.com/python-poetry/poetry/issues/6301) Poetry 1.5.1 cannot function on either the login server or the Slurm system. The fix is merged, but it will not land until 1.6.0.

### Python packages

Install the dependencies as mentioned in the [Installation](#installation) section on the login server. And then use the commands below to install the dependencies on the Slurm system with the Slurm shell.

```
poetry install --extras fujitsu-slurm
pip install -e .
```

To enable VS Code's language feature on the login server when editing the code that uses the MPI packages, execute `copy_packages_from_slurm.sh` on the login server after installing the dependencies on the Slurm system.
