# BART Experiments

## Environment set up
To set up the environments we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once you have miniconda installed, there is an `environment.yml` file in each folder to create an environment designed to run the tests in each one of the folders. These environments can be created from within each folder with:

```bash
conda env create -f environment.yml
conda activate bart-0.3.1
```

To run benchmarks you should do the following command
```bash
bash benchmark.sh
```


To see the plot of the performance profile you can run the following commands:
```bash
snakeviz $file
```

To see the plot of memory use you can run the following commands:
```bash
mprof plot $file
```


To update the local version of pymc-bart you can run:
```bash
pip uninstall --yes pymc-bart && pip install -e /Users/floyola/personal/pymc-bart
```
With conda:
```bash
conda install --name bart-0.3.1 -c local /Users/floyola/personal/pymc-bart
```

```bash
pip uninstall --yes pymc && pip install pymc==5.0.2
```

pip install -e /Users/floyola/personal/pymc-bart

## Troubleshooting
```
WARNING (pytensor.tensor.blas): Using NumPy C-API based implementation for BLAS functions.
```

sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran


conda install numpy scipy mkl
conda install theano
---pygpu
___ ultima version

conda install mkl
conda install mkl-service
conda install blas

conda install -c conda-forge aesara
conda install -c conda-forge numba
conda install -c conda-forge blas

Que funciono!
conda create -y -c conda-forge -n bart-3.0.2 python=3.9 pymc

    argparse
    matplotlib
    memory-profiler
    snakeviz
test:
python -c "import pymc; print(pymc.__version__)"

 from pymc.step_methods.compound import Competence

python -c "from pymc.step_methods.compound import Competence"

python -c "import pymc_bart; print(pymc_bart.__version__)"

conda activate bart-3.0.2

conda install -c conda-forge pytensor
conda install -c conda-forge numba
conda install -c conda-forge pathlib
conda install -c conda-forge matplotlib==3.5.2
conda install -c conda-forge pymc
to export env:
```bash
conda env export > environment.yml
```

conda env create -f env.yml
conda activate bart-0.3.2

pagina para tablaS:
https://tableizer.journalistopia.com/tableizer.php


https://github.com/nschloe/tuna

python -mcProfile -o program.prof cases_studies/import.py


tuna program.prof
python3 -X importtime -c "import pymc_bart" 2> bart_current.log
tuna bart_current.log
-> 5.2


python3 -X importtime -c "import pymc_bart" 2> bart_new.log
tuna bart_new.log

python -m black pymc_bart
