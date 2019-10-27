# Probabilistic Programming for Spectroscopic Data Analysis
> This repository contains Jupyter notebooks for probabilistic modeling of vibrational spectroscopic datasets. All models have been implemented using the Python probabilistic programming library PyMC3. To use the software, first setup your system by creating a virtual environment containing the required Python packages.

# System preparation

## 1. Linux System Setup (Ubuntu 18.04)

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

> To deactivate the virtual environment:
```
>> deactivate
```

### Install required Python packages

> Activate and cd into the created virtual environment. Install the following Python packages:
```
>> pip install --upgrade numpy scipy scikit-learn matplotlib pandas
jupyter seaborn pymc3 arviz graphviz
```

### Run Jupyter Notebook

> To run and experiment with the models, start a Jupyter Notebook server and open the .ipynb files containing the models inside the PPSDA/code/ directories:
```
>> jupyter notebook
```

## 2. Windows System Setup (Windows 7/10)

### Install Miniconda

> For Windows it is advised to first install the Miniconda environment ([https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)). After Miniconda installation, open a shell and create a new environment:
```
>> conda create --name PPSDA
```

> Enter the new environment by running:
```
>> conda activate PPSDA
```

> To exit the new environment enter:
```
>> conda deactivate
```

### Install required Python packages

> Activate the new environment and install the following Python packages: 
```
>> conda install numpy scipy scikit-learn matplotlib pandas jupyter
   seaborn pymc3
```

> Install the arviz and graphviz libraries:
```
>> conda install -c conda-forge arviz python-graphviz
```

### Run Jupyter Notebook

> To run and experiment with the models, start a Jupyter Notebook server and open the .ipynb files containing the models inside the PPSDA/code/ directories:
```
>> jupyter notebook
```

# Datasets

## Coffees
> The coffees dataset contains 56 FTIR samples of two coffee species, Arabica (29) and Robusta (27). The spectra were truncated to 800-2000 cm<sup>-1</sup>. The dataset was obtained from: [https://csr.quadram.ac.uk/example-datasets-for-download/](https://csr.quadram.ac.uk/example-datasets-for-download/)

## Olive Oils
> The olive oils dataset contains 120 FTIR samples originating from Spain (50), Italy (34), Greece (20) and Portugal (16), corresponding to four different classes. The spectra were truncated to 799-1897 cm<sup>-1</sup>. The dataset was obtained from: [https://csr.quadram.ac.uk/example-datasets-for-download/](https://csr.quadram.ac.uk/example-datasets-for-download/)

## Juices
> This dataset contains 983 FTIR samples originating from two classes of fresh fruit juices, non-strawberry (632) and strawberry (351). The spectra were truncated to 899-1802 cm<sup>-1</sup>. The dataset was obtained from: [https://csr.quadram.ac.uk/example-datasets-for-download/](https://csr.quadram.ac.uk/example-datasets-for-download/)
>