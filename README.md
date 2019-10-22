# Probabilistic Programming for Spectroscopic Data Analysis (PPSDA)
> This repository contains IPython notebooks for probabilistic modeling of vibrational spectroscopic datasets. All models have been implemented using the Python probabilistic programming library PyMC3.
>
> To use the software, first setup your system by creating a virtual Python environment containing the required packages.
>

## System Setup on Linux/Windows

> The setup instructions have been tested on Ubuntu 18.04 LTS and Windows 10.

## 1. Linux system setup (Ubuntu 18.04)

### Install virtualenv

> To avoid messing up your existing Python installation, first install virtualenv [(https://virtualenv.pypa.io/en/stable)](https://virtualenv.pypa.io/en/stable/):
>
```
>> pip install virtualenv
```

> Once installed, create a new virtual enviroment by running the command:

```
>> virtualenv PPSDA
```

> You can activate the virtual environment by running:
```
>> source PPSDA/bin/activate
```

> To deactivate the virtual environment, run:
```
>> deactivate
```

### Install required Python packages

> cd into the created virtual environment and install the following required Python packages:
```
>> pip install numpy scipy matplotlib pandas jupyter seaborn pymc3 arviz
```

### Run Jupyter Notebook

> To run and experiment with the models, start a Jupyter Notebook server and open the .ipynb files containing the models inside the PPSDA/code/ directories:
```
>> jupyter notebook
```

## 2. Windows system setup (Windows 10)

### Install Miniconda

> For Windows it is advised to first install the Miniconda environment ([https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)). After Miniconda installation, open a shell and create a new environment:
```
>> conda create --name PPSDA
```

> Enter the new environment by running:
```
>> conda activate PPSDA
```

> To exit the new environment type:
```
>> conda deactivate
```

### Install required Python packages

> Activate the new environment and install the following required Python packages: 
```
>> conda install numpy scipy matplotlib pandas jupyter seaborn pymc3 arviz 
```

### Run Jupyter Notebook

> To run and experiment with the models, start a Jupyter Notebook server and open the .ipynb files containing the models inside the PPSDA/code/ directories:
```
>> jupyter notebook
```

