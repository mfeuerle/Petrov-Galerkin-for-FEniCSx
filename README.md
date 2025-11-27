# Petrov-Galerkin addon for FEniCSx

The documentation can be found [here](https://mfeuerle.github.io/Petrov-Galerkin-for-FEniCSx/).

For installation run

```bash
$ conda env create -f environment.yml
$ conda activate pgfenicsx
$ pip install .
```

for an editable installation

```bash
$ pip install .
```

To build the docs also run

```bash
$ conda env update -f docs/environment-sphinx.yml
$ cd docs
$ make html
```
