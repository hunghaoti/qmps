# Representing, optimizing and evolving matrix product states on NISQ devices

## Installation
Clone this somewhere. Run the commands below from this folder.

```
pip install -e .
pip install --use-pep517 git+https://github.com/fergusbarratt/xmps.git@master
```

## Prepare

Copy `qmps/loschmidts/real_dev_submit/20230711/src/config.example.ini` to `qmps/loschmidts/real_dev_submit/20230711/src/config.ini` and put your IBM API token into `config.ini`. Get your IBM API token at the [IBM Quantum Account page](https://quantum-computing.ibm.com/account).
