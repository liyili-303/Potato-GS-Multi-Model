library(BGLR)

# ===================== =====================
geno <- as.matrix(read.csv("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/geno_for_R.csv", header=FALSE))
pheno <- read.table("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/maspot_qc_pheno.txt", header=FALSE)
# =================================================================

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno  


seeds <- c(42,123, 2024, 66, 999)
model_map <- list(GBLUP = "GBLUP")
results <- data.frame()

# ===================== =====================
for (seed in seeds) {
  set.seed(seed)
  cat("\n=============================================\n")
  cat("============= 种子：", seed, " =============\n")
  
  folds <- sample(rep(1:5, length.out = length(y)))
  
  for (model_name in names(model_map)) {
    cat("\n📌 模型：", model_name, "\n")
    y_pred <- rep(NA, length(y))
    
    for (k in 1:5) {
      t0 <- Sys.time()
      test_idx <- which(folds == k)
      train_idx <- which(folds != k)
      
      # 1. 数据拆分
      Xtr <- X[train_idx, ]
      ytr <- y[train_idx]
      Xte <- X[test_idx, ]
      yte <- y[test_idx]
      n_tr <- nrow(Xtr)
      n_te <- nrow(Xte)
      
   
      Xtr_s <- scale(Xtr)
      G <- tcrossprod(Xtr_s) / ncol(Xtr_s)
      
     
      Xte_s <- scale(Xte, center=attr(Xtr_s,"scaled:center"), scale=attr(Xtr_s,"scaled:scale"))
      K_te_tr <- tcrossprod(Xte_s, Xtr_s) / ncol(Xtr_s)
      
     
      ETA <- list(list(X = G, model = "BRR"))
      fit <- BGLR(y = ytr, ETA = ETA, nIter = 6000, burnIn = 2000, verbose = FALSE)
      
      
      yte_pred <- K_te_tr %*% fit$ETA[[1]]$b + fit$mu
      y_pred[test_idx] <- as.numeric(yte_pred)
      
      
      rmse_f <- sqrt(mean((yte - yte_pred)^2))
      r2_f <- cor(yte, yte_pred)^2
      time_f <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      
      cat(sprintf("  折%d | RMSE=%.4f | R²=%.4f | 耗时=%.2fs\n", k, rmse_f, r2_f, time_f))
    }
    
    
    total_rmse <- sqrt(mean((y - y_pred)^2))
    total_r2 <- cor(y, y_pred)^2
    results <- rbind(results, data.frame(seed=seed, model=model_name, RMSE=total_rmse, R2=total_r2))
    cat(sprintf("✅ 完成 | 总RMSE=%.4f | R²=%.4f\n", total_rmse, total_r2))
  }
}

# 输出最终结果
cat("\n\n==================== GBLUP 最终结果 ====================\n")
print(results)