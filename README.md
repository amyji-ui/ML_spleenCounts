# ML_SPLEENCOUNTS

A machine-learning project for classifying spleen single-cell data using several supervised learning models. The workflow includes preprocessing, highly variable gene (HVG) selection, PCA-based dimensionality reduction, model training, cross-validation, and evaluation figure generation.

Methods implemented from scratch: PCA, SVM and KNN.

# Data
unzip the data.zip file and leave the data folder in the root directory. See Directory tree for reference.

## Directory Tree

```text
ML_SPLEENCOUNTS/
├── .venv/
├── data/
├── output/
├── qmetric/
│   ├── knn_evaluation.png
│   ├── multinomial_logistic_evaluation.png
│   ├── random_forest_evaluation.png
│   └── svm_evaluation.png
├── script/
│   ├── __pycache__/
│   ├── fit_scratch_svm.py
│   ├── HVG.py
│   ├── KFoldCrossVal.py
│   ├── kNN_pack.py
│   ├── kNN.py
│   ├── mlr.py
│   ├── pca_deprecated.py
│   ├── pca_simplified.py
│   ├── preprocessing.py
│   ├── prf.py
│   ├── random_forest.py
│   ├── SVM_pk.py
│   └── SVM_scratch.py
├── .gitignore
├── main.py
└── README.md
```

## Project Overview

This project compares multiple classification models on spleen single-cell data. The main goal is to evaluate how well different machine-learning approaches can predict cell labels from gene-expression features.

The project includes implementations or scripts for:

- k-nearest neighbors, or KNN
- Multinomial logistic regression
- Random forest
- Support vector machine, or SVM
- SVM implementation from scratch
- PCA-based feature reduction
- HVG-based feature selection
- K-fold cross-validation
- Precision, recall, and F1-score evaluation

## Repository Structure

### `data/`

Stores input datasets used for model training and testing. Raw or processed gene-expression matrices should be placed here.

### `output/`

Stores generated intermediate files or processed outputs from preprocessing, feature selection, PCA, and model execution.

### `qmetric/`

Stores model evaluation figures. Current evaluation plots include:

- `knn_evaluation.png`
- `multinomial_logistic_evaluation.png`
- `random_forest_evaluation.png`
- `svm_evaluation.png`

These figures summarize model performance using metrics such as precision, recall, F1-score, confusion matrices, or precision-recall curves depending on the script output.

### `script/`

Contains the main Python scripts used in the analysis pipeline.

| File | Purpose |
|---|---|
| `preprocessing.py` | Loads and preprocesses the dataset before modeling. |
| `HVG.py` | Selects highly variable genes to reduce the feature space. |
| `pca_simplified.py` | Applies PCA for dimensionality reduction. |
| `pca_deprecated.py` | Older PCA implementation retained for reference. |
| `KFoldCrossVal.py` | Performs K-fold cross-validation. |
| `kNN.py` | Runs KNN classification. |
| `kNN_pack.py` | Packaged or helper version of the KNN workflow. |
| `mlr.py` | Runs multinomial logistic regression. |
| `random_forest.py` | Runs random forest classification. |
| `SVM_pk.py` | Runs package-based SVM classification. |
| `SVM_scratch.py` | Implements SVM training logic from scratch. |
| `fit_scratch_svm.py` | Fits and evaluates the scratch SVM model. |
| `prf.py` | Computes or plots precision, recall, and F1-score metrics. |

### `main.py`

Main entry point for running the project pipeline.

### `.venv/`

Local Python virtual environment. Set up if needed.

## Basic Workflow

1. Place the input data files in the `data/` directory.
2. Run preprocessing to clean and format the dataset.
3. Apply HVG selection and/or PCA to reduce dimensionality.
4. Train classification models using the scripts in `script/`.
5. Evaluate each model using cross-validation and held-out test data.
6. Save model evaluation figures in `qmetric/`.

## Running the Project

From the project root directory, run:

```bash
python main.py
```

Individual model scripts can also be run from the `script/` directory, depending on the desired analysis.

Example:

```bash
python script/random_forest.py
python script/SVM_pk.py
python script/kNN.py
```

## Notes

Because single-cell gene-expression datasets often contain many more genes than cells, dimensionality reduction and feature selection are important. This project uses HVG selection and PCA to reduce the number of input features before model training. Regularization should also be considered for models such as logistic regression and SVM to reduce overfitting and improve generalization, but is not done in this project due to the simplicity of our data.

## Suggested Dependencies

The project likely requires the following Python packages:

```text
numpy
pandas
scikit-learn
matplotlib
seaborn
scanpy
anndata
```

Install dependencies with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scanpy anndata
```

## Outputs

Final model evaluation plots are saved in:

```text
qmetric/
```

These plots can be used directly in reports or presentations to compare model performance.

