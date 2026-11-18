DATA_DIR  <- "C:/Users/liyili/Desktop/liqiang/MASPOT_GWAS/data"
GENO_FILE <- file.path(DATA_DIR, "geno_for_R.csv")
PHENO_FILE <- file.path(DATA_DIR, "maspot_qc_pheno.txt")

# ---- 1. 读取数据 ------------------------------------------------------------
library(BGLR)

geno <- as.matrix(read.csv(GENO_FILE, header = FALSE))
pheno <- read.table(PHENO_FILE, header = FALSE)

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno

# ---- 2. 评估设置 ------------------------------------------------------------
seeds <- c(42, 66, 123, 999, 2024)
n_folds <- 5
results <- data.frame()

# R2 = squared Pearson correlation coefficient (与论文公式 4 口径一致)
r2_score <- function(y_obs, y_pred) cor(y_obs, y_pred)^2

# ---- 3. 重复交叉验证主循环 --------------------------------------------------
for (seed in seeds) {
  set.seed(seed)
  cat("\n=============================================\n")
  cat("============= 种子:", seed, "=============\n")

  folds <- sample(rep(1:n_folds, length.out = length(y)))
  y_pred <- rep(NA, length(y))

  for (k in 1:n_folds) {
    t0 <- Sys.time()
    test_idx  <- which(folds == k)
    train_idx <- which(folds != k)
    Xtr <- X[train_idx, ]
    ytr <- y[train_idx]
    Xte <- X[test_idx, ]

    ETA <- list(list(X = Xtr, model = "BRR"))
    fit <- BGLR(y = ytr, ETA = ETA,
                nIter = 6000, burnIn = 2000, verbose = FALSE)

    y_pred[test_idx] <- Xte %*% fit$ETA[[1]]$b + fit$mu

    rmse_f <- sqrt(mean((y[test_idx] - y_pred[test_idx])^2))
    r2_f   <- r2_score(y[test_idx], y_pred[test_idx])
    cat(sprintf("  fold %d | RMSE=%.4f | R2=%.4f | %.2fs\n",
                k, rmse_f, r2_f, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }

  total_rmse <- sqrt(mean((y - y_pred)^2))
  total_r2   <- r2_score(y, y_pred)
  results <- rbind(results, data.frame(seed = seed, model = "RR-BLUP",
                                       RMSE = total_rmse, R2 = total_r2))
  cat(sprintf("✅ 完成 | 总 RMSE=%.4f | R2=%.4f\n", total_rmse, total_r2))
}

# ---- 4. 汇总 ----------------------------------------------------------------
cat("\n\n==================== RR-BLUP 最终结果 ====================\n")
print(results)
cat(sprintf("\nMean R2 = %.4f ± %.4f\n", mean(results$R2), sd(results$R2)))
cat(sprintf("Mean RMSE = %.4f ± %.4f\n", mean(results$RMSE), sd(results$RMSE)))
