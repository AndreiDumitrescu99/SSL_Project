# Abusive Language Detection in Social Media
## Ionescu Diana & Dumitrescu Andrei

### Overview
This repository contains the final code and documentation for the SSL Project. <br>
The topic of the project is Abusive Language Detection in Social Media. <br>
The project offers multiple baselines and State-of-the-Art Models. <br>
As requirments for running the experiments you need the following APIs: TensorFlow, Numpy, PyTorch, HuggingFace Transformers, Sklearn, Pandas, NLTK.

### Repository Structure
The structure of the repository is the following: <br>
    * `docs` folder: contains all the documentation presented during this semester <br>
    * `datasets` folder: contains all the datasets used for this project <br>
    * `plots` folder: contains some graphs plotted during the first iteration experiments <br>
    * `baselines.py`, `first_iteration.py`, `second_iteration.py` files: contain the code for training and evaluating 
        the baseline / first iteration / second iteration <br>
    * `torch_dataset.py` file: contains a custom dataset implementation <br>
    * `data_reader.py`, `preprocess.py` files: contain auxiliary functions used for data reading and processing <br>
    * `utils.py` file: contains auxiliary functions used for plotting graphs <br>
    * `prediction.py` file: contains code for predicting on different datasets <br>

### Use Case
Each of the files `baselines.py`, `first_iteration.py`, `second_iteration.py` contain functions that follow the template: `run_X_experiments(...)`. <br>
To train and evaluate the second iteration models simply run:
```bash
python second_iteration.py
```
The models from the baselines and first iteration will be saved in a folder called `models`. <br>
The models from the second iteration will be saved in a folder called `models_torch`. <br>
Make sure that these folders exist! <br>
To run the prediction demo simply run:
```bash
python prediction.py
```
You can change the global parameters to run on any dataset you want.