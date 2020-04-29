# General

This repository provides code and examples for generating nearest counterfactual explanations and minimal consequential interventions. The following papers are supported:

- [2017.06 Feature Tweaking](https://arxiv.org/abs/1706.06691) (4c691b4 @ https://github.com/upura/featureTweakPy)
- [2019.01 Actionable Recourse](https://arxiv.org/pdf/1809.06514) (9387e6c @ https://github.com/ustunb/actionable-recourse)
- [2019.07 Minimum  Observable](https://arxiv.org/abs/1907.04135)
- [2019.05 MACE](https://arxiv.org/abs/1905.11190)
- [2020.02 MINT](https://arxiv.org/abs/2002.06278)



# Code Pre-requisites

First,
```console
$ git clone https://github.com/amirhk/mace.git
$ pip install virtualenv
$ cd mace
$ virtualenv -p python3 _venv
$ source _venv/bin/activate
$ pip install -r pip_requirements.txt
$ pysmt-install --z3 --confirm-agreement
```


Then refer to
```console
$ python batchTest.py  --help
```

and run as follows
```console
$ python batchTest.py -d *dataset* -m *model* -n *norm* -a *approach* -b 0 -s *numSamples*
```

For instance, you may run
```console
$ python batchTest.py -d adult -m lr -n zero_norm -a AR -b 0 -s 1
$ python batchTest.py -d credit -m mlp -n one_norm -a MACE_eps_1e-3 -b 0 -s 1
$ python batchTest.py -d german -m tree -n two_norm -a MINT__eps_1e-3 -b 0 -s 1
$ python batchTest.py -d mortgage -m forest -n infty_norm -a MINT__eps_1e-3 -b 0 -s 1
```

Finally, view the results under the _experiments folder.



# Specific considerations for _minimal interventions_

For mortgage data, where a causal structure governs the world, AND all variables
are actionable and mutable, we should expect to see `int_dist <= ? >= cfe_dist`,
but `cfe_dist <= scf_dist`. You can assert this by running the following:

```console
$ python batchTest.py -d mortgage -m lr -n one_norm -a MINT_eps_1e-5 MACE_eps_1e-5 -b 0 -s 10
```

Then you can compare the distances resulting fron MACE and MINT as outputted in the console. Do make sure to run `batchTest.py` with `loadData.loadDataset(load_from_cache = True)` so that MACE and MINT use the same data and the resulting comparison is fair.



# Using git-hooks script for sanity checking

There is a `pre-push` script under `_hooks/` which can be used to check MACE under different setups.
Specifically, it checks for successfully running of the code and the closeness of the generated CFEs
to the previously-saved (approximately) optimal ones. You can either manually call the script from MACE root directory by
`_hooks/pre-push` or place it under your local `.git/hooks/` directory to run automatically before every push.
In this case, please remember to give it the required permissions:

```console
$ chmod +x .git/hooks/pre-push
```

