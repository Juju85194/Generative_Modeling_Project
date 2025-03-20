# Generative_Modeling_Project

Report containing the code and the report of the projet 5 [PSEUDOINVERSE-GUIDED DIFFUSION MODELS FOR INVERSE PROBLEMS](https://openreview.net/pdf?id=9_gsMA8MRKQ) of *Generative Models for image* @ MVA (B. GALERNE & A. LECLAIRE)

## Structure

```
Generative_modeling_project/
├── README.md
├── init.ipynb
├── setup.py
├── data/
│   └── __init__.py
├── figs/
│   └── __init__.py
├── ntbk/
│   ├── __init__.py
│   ├── evaluator.ipynb
│   └── tp_6.ipynb
├── src/
│   ├── __init__.py
│   ├── ddpm.py
│   ├── eval.py
│   ├── h_fcn.py
│   ├── h_utils.py
│   ├── pigdm.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test.ipynb
```

* src: contains all the function of the project
* figs: contains some outputs of the model
* tests: sandbox with jupyter notebooks
* ntbk: notebooks to present our main results and how to reproduce them
* data: containing all the data and the pre trained models

## Get Start

First, please run:
```bash
pip install -e .
```

from a terminal or a notebook. Or simply run the notebook ``init.ipynb`` to download the packages, the pre-trained models and the data

#TODO: faire les scripts pour run les experiences, insérer un dossier report dans lequel on mettra les pdf/slides etc... Commenter certaines parties dans src 
