# mace

```console
$ git clone https://github.com/amirhk/mace.git
$ pip install virtualenv
$ cd mace
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip install -r pip_requirements.txt
$ pysmt-install --z3 --confirm-agreement
```

Then run
```console
$ python batchTest.py -d *dataset* -m *model* -n *norm* -a *approach* -b 0 -s *numSamples*
```
