#!/bin/bash

# This script copies the packages only available on Fujitsu's Slurm system to
# the login server to enable some editor's language features on the login
# server when editing the code that uses the MPI packages.

arm_venv_path=$(salloc poetry env info -p 2>/dev/null)
x64_venv_path=$(poetry env info -p)

# Assumes there isonly one folder under lib/. Maybe it will be better to use
# `python -c 'import site; print(site.getsitepackages())'` to get the exact
# site path.
arm_site="$arm_venv_path/lib/python*/site-packages"
x64_site="$x64_venv_path/lib/python*/site-packages"

eval cp -r "$arm_site/qiskit_qulacs*" "$x64_site/"
eval cp -r "$arm_site/mpi*" "$x64_site/"
