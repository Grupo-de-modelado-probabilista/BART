# BART Experiments

## Environment set up
To set up the environments we used [miniconda](https://docs.conda.io/en/latest/miniconda.html).

Once you have miniconda installed, there is an `environment.yml` file in each folder to create an environment designed to run the tests in each one of the folders. These environments can be created from within each folder with:

```bash
conda env create -f environment.yml
conda activate bart-0.3.0
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
