# Genomic Selection Models for Tetraploid Potato

This repository contains the genotype and phenotype data together with the R and Python scripts
used in our study to implement and compare unimodal and multimodal genomic selection (GS) models
for tetraploid potato (*Solanum tuberosum*, 4x). All models are evaluated by five-fold
cross-validation, repeated over five random seeds (**42, 123, 2024, 66, 999**), so that predictive
accuracy can be reported as a mean over seeds.

## Repository structure

```
.
├── README.md
├── BayesABC.R
├── BRR_RRBLUP.R
├── GBLUP.R
├── ElasticNet.R
├── KRR.R
├── XGBoost_ISSA.py
└── data/
    ├── geno_for_R.csv.gz.01 ... .07   # gzip genotype matrix, split into 7 parts (~47 MB total)
    └── y.csv                          # phenotype table with header
```

## Scripts

| File | Model | Implementation | Language |
| --- | --- | --- | --- |
| `BayesABC.R` | BayesA (BayesB / BayesC by editing `model_map`) | `BGLR` | R |
| `BRR_RRBLUP.R` | Bayesian Ridge Regression (RR-BLUP) | `BGLR` | R |
| `GBLUP.R` | Genomic Best Linear Unbiased Prediction | `BGLR` | R |
| `ElasticNet.R` | Elastic Net (5-fold CV, `glmnet`) | `glmnet` | R |
| `KRR.R` | Kernel Ridge Regression with an RBF kernel and hyperparameter search | base R | R |
| `XGBoost_ISSA.py` | XGBoost with hyperparameters tuned by an improved sparrow search algorithm (ISSA) | `xgboost` | Python |

## Data

* **`data/geno_for_R.csv.gz.01` … `.07`** — genotype/marker matrix, individuals in rows and SNPs in
  columns, no header, additive dosage coding (values between 0 and 1). The uncompressed matrix is
  about 177 MB; the gzip archive is about 47 MB and is **split into 7 parts** to fit the repository
  upload limit. Reassemble the parts, then decompress:

  Linux / macOS:

  ```bash
  cat data/geno_for_R.csv.gz.0* > geno_for_R.csv.gz
  gzip -d geno_for_R.csv.gz          # produces geno_for_R.csv
  ```

  Windows (Command Prompt):

  ```cmd
  copy /b data\geno_for_R.csv.gz.01+data\geno_for_R.csv.gz.02+data\geno_for_R.csv.gz.03+data\geno_for_R.csv.gz.04+data\geno_for_R.csv.gz.05+data\geno_for_R.csv.gz.06+data\geno_for_R.csv.gz.07 geno_for_R.csv.gz
  ```

  Then extract `geno_for_R.csv.gz` with 7-Zip (or `gzip -d` if Git Bash is available). The resulting
  `geno_for_R.csv` is the file read by every script.

* **`data/y.csv`** — phenotype table with a header row. The identifier columns are
  `Cultivar, Panel, Mother, Father`, followed by the measured traits
  (`Yield [hkg/ha]`, `Dry matter content [%]`, `Starch content [%]`,
  `Chipping quality [scale 1-9]`, `Length ...`).

Note: the scripts were originally run against a phenotype file (`maspot_qc_pheno.txt`) in which the
response variable is the **last column** (`y <- as.numeric(pheno[, ncol(pheno)])`). When using
`y.csv`, drop the four identifier columns and select the trait you want to model as the response,
for example:

```r
pheno <- read.csv("data/y.csv", header = TRUE, row.names = 1)
y <- as.numeric(pheno[["Yield [hkg/ha]"]]); y <- y[!is.na(y)]  # response vector
geno <- as.matrix(read.csv("data/geno_for_R.csv", header = FALSE))
```

## Requirements

* R (>= 4.0) with the packages: `BGLR`, `glmnet`, `data.table`
* Python (>= 3.9) with the packages: `numpy`, `pandas`, `xgboost`, `scikit-learn`

## Usage

1. Decompress the genotype file (see above) so that `data/geno_for_R.csv` exists.
2. Each script contains two hard-coded absolute paths, for example:

   ```r
   geno  <- as.matrix(read.csv(".../data/geno_for_R.csv", header = FALSE))
   pheno <- read.table(".../data/maspot_qc_pheno.txt", header = FALSE)
   ```

   Replace them with the paths on your own machine before running.
3. Run the script (R: `source("BRR_RRBLUP.R")`; Python: `python XGBoost_ISSA.py`).

## Notes

* The five seeds are defined near the top of every script
  (`seeds <- c(42, 123, 2024, 66, 999)`); changing them reproduces the same protocol with
  different cross-validation partitions.
* Predictive accuracy is summarized per seed and model in a results data frame written by each
  script.

## Contact

For questions about the data or the scripts, please contact the corresponding author of the paper.
