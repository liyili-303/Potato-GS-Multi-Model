# Genomic Selection Models for Tetraploid Potato

This repository contains the genotype and phenotype data together with the R and Python
scripts used in our study to implement and compare unimodal and multimodal genomic
selection (GS) models for tetraploid potato (*Solanum tuberosum*, 4x). All models are
evaluated by five-fold cross-validation, repeated over five random seeds (42, 123, 2024,
66, 999), so that predictive accuracy can be reported as a mean over seeds.

## Repository contents

| File | Model |
| --- | --- |
| `BayesABC.R` | Bayes A / Bayes B / Bayes C (BGLR) |
| `BRR_RRBLUP.R` | Bayesian Ridge Regression / RR-BLUP (BGLR) |
| `ElasticNet.R` | Elastic Net (glmnet, alpha and lambda tuned by grid search) |
| `GBLUP.R` | Genomic BLUP (BGLR) |
| `KRR.R` | Kernel Ridge Regression (Gaussian kernel, sigma and lambda tuned by grid search) |
| `XGBoost_ISSA.py` | XGBoost (tree and linear booster) tuned by the improved sparrow search algorithm (ISSA) |

## Data

`data/geno_for_R.csv.gz.01` ... `data/geno_for_R.csv.gz.07` — the genotype matrix
(34106 SNP markers, one individual per row, no header), stored as a gzip-compressed CSV
split into seven parts so that each part stays below GitHub's per-file upload limit.
Reassemble the parts before use.

Linux / macOS:

```bash
cat data/geno_for_R.csv.gz.0* > geno_for_R.csv.gz
gunzip geno_for_R.csv.gz
```

Windows PowerShell:

```powershell
Get-Content data\geno_for_R.csv.gz.0* -Encoding Byte -ReadCount 0 |
    Set-Content geno_for_R.csv.gz -Encoding Byte
```

In R: `geno <- as.matrix(read.csv("geno_for_R.csv", header = FALSE))`.

`data/maspot_qc_pheno.txt` — tab-separated phenotype file with two columns: the
clone/individual identifier and the phenotype value. The scripts use the last column as
the response variable (`y <- as.numeric(pheno[, ncol(pheno)])`), and the row order
matches the row order of the genotype matrix.

## How to run

Every script starts with the same three path constants:

```r
DATA_DIR   <- "path/to/data"
GENO_FILE  <- file.path(DATA_DIR, "geno_for_R.csv")
PHENO_FILE <- file.path(DATA_DIR, "maspot_qc_pheno.txt")
```

Set `DATA_DIR` to the folder that holds the two data files, then run the script from top
to bottom. Cross-validation folds are generated inside each script from the same five
seeds, so the scripts can be run independently and the results compared directly.

## Note on the XGBoost script

`XGBoost_ISSA.py` reads a pre-processed, imputed marker matrix in NumPy format
(`maspot_qc_imputed.npy`, about 200 MB), which is not included here. If you only have the
CSV matrix, convert it once and point `GENO_NPY` at the result:

```python
import numpy as np
np.save("maspot_qc_imputed.npy", np.loadtxt("geno_for_R.csv", delimiter=","))
```

## Requirements

- R (>= 4.2) with the packages `BGLR` and `glmnet`
- Python (>= 3.9) with `numpy`, `pandas`, `scikit-learn` and `xgboost`

## Contact

Li Yili (GitHub: liyili-303), Yunnan Agricultural University — 2451245724@qq.com
