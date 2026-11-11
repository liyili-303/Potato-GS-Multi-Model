# 仅加载BGLR
library(BGLR)

# ===================== =====================
geno <- as.matrix(read.csv("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/geno_for_R.csv", header=FALSE))
pheno <- read.table("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/maspot_qc_pheno.txt", header=FALSE)
# =================================================================

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno  


seeds <- c(42,123, 2024, 66, 999)

# 
model_map <- list(
  RR_BLUP   = "BRR" 
)

results <- data.frame()

# ===================== 主循环=====================
for (seed in seeds) {
  set.seed(seed)
  cat("\n=============================================\n")
  cat("============= 种子：", seed, " =============\n")
  
  # 5折划分
  folds <- sample(rep(1:5, length.out = length(y)))
  
  for (model_name in names(model_map)) {
    bglr_mod <- model_map[[model_name]]
    cat("\n📌 模型：", model_name, "\n")
    y_pred <- rep(NA, length(y))
    
    for (k in 1:5) {
      t0 <- Sys.time()
      # 训练/测试集
      test_idx <- which(folds == k)
      train_idx <- which(folds != k)
      Xtr <- X[train_idx, ]
      ytr <- y[train_idx]
      Xte <- X[test_idx, ]
      
      # 你最早原版的标准写法
      ETA <- list(list(X = Xtr, model = bglr_mod))
      fit <- BGLR(y = ytr, ETA = ETA, nIter = 6000, burnIn = 2000, verbose = FALSE)
      
      y_pred[test_idx] <- Xte %*% fit$ETA[[1]]$b + fit$mu
      
      # 计算指标（不变）
      yte <- y[test_idx]
      rmse_f <- sqrt(mean((yte - y_pred[test_idx])^2))
      r2_f <- cor(yte, y_pred[test_idx])^2
      
      # 
      time_f <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      
      cat(sprintf("  折%d | RMSE=%.4f | R²=%.4f | 耗时=%.2fs\n", k, rmse_f, r2_f, time_f))
    }
    
    # 汇总结果
    total_rmse <- sqrt(mean((y - y_pred)^2))
    total_r2 <- cor(y, y_pred)^2
    results <- rbind(results, data.frame(seed=seed, model=model_name, RMSE=total_rmse, R2=total_r2))
    cat(sprintf("✅ 完成 | 总RMSE=%.4f | R²=%.4f\n", total_rmse, total_r2))
  }
}

# 输出最终结果
cat("\n\n==================== BRR 最终结果 ====================\n")
print(results)
   