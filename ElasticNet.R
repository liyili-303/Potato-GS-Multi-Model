DATA_DIR  <- "C:/Users/liyili/Desktop/liqiang/MASPOT_GWAS/data"
GENO_FILE <- file.path(DATA_DIR, "geno_for_R.csv")
PHENO_FILE <- file.path(DATA_DIR, "maspot_qc_pheno.txt")

# ---- 1. 加载包与数据 --------------------------------------------------------
library(glmnet)

geno <- as.matrix(read.csv(GENO_FILE, header = FALSE))
pheno <- read.table(PHENO_FILE, header = FALSE)

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno

# ---- 2. 评估设置 ------------------------------------------------------------
seeds <- c(42, 66, 123, 999, 2024)
n_folds <- 5
alpha_candidates <- c(0.1, 0.5, 0.7, 0.9, 0.95, 0.99)

records_opt <- list()   # 优化组结果
records_def <- list()   # 默认基线组结果

r2_score <- function(y_obs, y_pred) cor(y_obs, y_pred)^2

# ---- 3. 重复交叉验证主循环 --------------------------------------------------
for (seed in seeds) {
  set.seed(seed)
  folds <- sample(rep(1:n_folds, length.out = nrow(X)))

  for (fold in 1:n_folds) {
    train_idx <- which(folds != fold)
    test_idx  <- which(folds == fold)
    X_train <- X[train_idx, ]
    y_train <- y[train_idx]
    X_test  <- X[test_idx, ]
    y_test  <- y[test_idx]

    # ----- 3.1 优化组: 外层遍历 α, 内层 5 折 CV 选 λ -----
    best_rmse <- Inf
    best_fit  <- NULL
    best_alpha <- NA
    best_lambda <- NA

    for (alpha_val in alpha_candidates) {
      cvfit <- tryCatch(
        cv.glmnet(X_train, y_train, alpha = alpha_val, nfolds = 5,
                  standardize = TRUE, type.measure = "mse"),
        error = function(e) NULL
      )
      if (is.null(cvfit)) next
      rmse_val <- sqrt(min(cvfit$cvm))
      if (rmse_val < best_rmse) {
        best_rmse   <- rmse_val
        best_fit    <- cvfit
        best_alpha  <- alpha_val
        best_lambda <- cvfit$lambda.min
      }
    }

    y_pred_opt <- predict(best_fit, newx = X_test, s = "lambda.min")
    records_opt[[length(records_opt) + 1]] <- data.frame(
      seed = seed, fold = fold,
      rmse = sqrt(mean((y_test - y_pred_opt)^2)),
      r2   = r2_score(y_test, y_pred_opt),
      best_alpha = best_alpha, best_lambda = best_lambda
    )

    # ----- 3.2 默认基线组: α=1 (pure Lasso), 模型默认 λ -----
    def_fit <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 5,
                         standardize = TRUE, type.measure = "mse")
    y_pred_def <- predict(def_fit, newx = X_test, s = "lambda.min")
    records_def[[length(records_def) + 1]] <- data.frame(
      seed = seed, fold = fold,
      rmse = sqrt(mean((y_test - y_pred_def)^2)),
      r2   = r2_score(y_test, y_pred_def),
      best_alpha = 1, best_lambda = def_fit$lambda.min
    )

    cat(sprintf("种子 %d | 第 %d 折 | 优化 R2=%.4f RMSE=%.4f (α=%.2f, λ=%.6f) | 默认 R2=%.4f RMSE=%.4f\n",
                seed, fold,
                r2_score(y_test, y_pred_opt), sqrt(mean((y_test - y_pred_opt)^2)),
                best_alpha, best_lambda,
                r2_score(y_test, y_pred_def), sqrt(mean((y_test - y_pred_def)^2))))
  }
}

# ---- 4. 汇总 ----------------------------------------------------------------
results_opt <- do.call(rbind, records_opt)
results_def <- do.call(rbind, records_def)

cat("\n==================== Elastic Net 优化组 25 次结果 ====================\n")
print(results_opt)
cat(sprintf("\n优化组  Mean R2 = %.4f ± %.4f | Mean RMSE = %.4f ± %.4f\n",
            mean(results_opt$r2), sd(results_opt$r2),
            mean(results_opt$rmse), sd(results_opt$rmse)))

cat("\n==================== Elastic Net 默认基线组 25 次结果 ====================\n")
print(results_def)
cat(sprintf("\n默认组  Mean R2 = %.4f ± %.4f | Mean RMSE = %.4f ± %.4f\n",
            mean(results_def$r2), sd(results_def$r2),
            mean(results_def$rmse), sd(results_def$rmse)))

write.csv(results_opt, "elastic_net_optimized_results.csv", row.names = FALSE)
write.csv(results_def, "elastic_net_default_results.csv", row.names = FALSE)
