# Representing, optimizing and evolving matrix product states on NISQ devices

## Installation

1. Clone this somewhere.
2. Install [poetry](https://python-poetry.org/docs/#installation)
3. Run the commands below from this folder.

```
poetry install
pip install -e .
```

If your are going to run the script on Fujitsu's Slurm system, run the commands below instead.

```
poetry install --extras fujitsu-slurm
pip install -e .
```

## Prepare

Copy `qite/src/config.example.ini` to `qite/src/config.ini` and put your IBM API token into `config.ini`. Get your IBM API token at the [IBM Quantum Account page](https://quantum-computing.ibm.com/account).
