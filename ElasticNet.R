

# ==================== 1. 加载必要包 ====================
library(glmnet)      # 弹性网络
library(data.table)  # 快速读取

# ==================== 2. 读取数据 ====================
geno <- as.matrix(read.csv("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/geno_for_R.csv", header=FALSE))
pheno <- read.table("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/maspot_qc_pheno.txt", header=FALSE)
# =================================================================

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno

# ==================== 3. 评估参数 ====================
seeds <- c(42,123, 2024, 66, 999)
n_folds <- 5
records <- list()   # 存储每折结果

# ==================== 4. 核心交叉验证循环 ====================
for (seed in seeds) {
  set.seed(seed)
  # 生成 5 折划分的索引
  folds <- sample(rep(1:n_folds, length.out = nrow(X)))
  
  for (fold in 1:n_folds) {
    train_idx <- which(folds != fold)
    test_idx  <- which(folds == fold)
    
    X_train <- X[train_idx, ]
    y_train <- y[train_idx]
    X_test  <- X[test_idx, ]
    y_test  <- y[test_idx]
    
    # ----- 弹性网络模型，内层交叉验证选择 alpha 和 lambda -----
    # glmnet 中：alpha 是弹性网络混合参数（0=岭回归，1=LASSO）
    # 我们搜索指定的 l1_ratio 范围
    alpha_candidates <- c(0.1, 0.5, 0.7, 0.9, 0.95, 0.99)
    best_rmse <- Inf
    best_fit <- NULL
    
    for (alpha_val in alpha_candidates) {
      # cv.glmnet 自动对 lambda 序列做交叉验证
      cvfit <- tryCatch(
        cv.glmnet(X_train, y_train, alpha = alpha_val, nfolds = 3, 
                  standardize = TRUE, type.measure = "mse"),
        error = function(e) NULL
      )
      if (is.null(cvfit)) next
      
      # 取最小 MSE 对应的 lambda
      rmse_val <- sqrt(min(cvfit$cvm))
      if (rmse_val < best_rmse) {
        best_rmse <- rmse_val
        best_fit <- cvfit
        best_alpha <- alpha_val
        best_lambda <- cvfit$lambda.min
      }
    }
    
    # ----- 在测试集上预测 -----
    y_pred <- predict(best_fit, newx = X_test, s = "lambda.min")
    
    # ----- 计算指标 -----
    rmse_test <- sqrt(mean((y_test - y_pred)^2))
    cor_val <- cor(y_test, y_pred)
    r2 <- cor_val^2
    
    records[[length(records) + 1]] <- data.frame(
      seed = seed,
      fold = fold,
      rmse = rmse_test,
      r2 = r2,
      best_alpha = best_alpha,
      best_lambda = best_lambda
    )
    
    cat(sprintf("种子 %d | 第 %d 折 | RMSE: %.4f | R²: %.4f | 最优 alpha: %.2f, lambda: %.6f\n",
                seed, fold, rmse_test, r2, best_alpha, best_lambda))
  }
}

# ==================== 5. 结果汇总 ====================
results <- do.call(rbind, records)
print("\n==================== 全部 20 次评估结果 ====================")
print(results)

# 各种子汇总
cat("\n==================== 各种子汇总 ====================\n")
aggregate(cbind(rmse, r2) ~ seed, data = results, 
          function(x) c(mean = mean(x), sd = sd(x)))

# 全局汇总
cat("\n==================== 弹性网络 (glmnet) 最终性能 ====================\n")
cat(sprintf("RMSE 均值: %.4f ± %.4f\n", mean(results$rmse), sd(results$rmse)))
cat(sprintf("R² 均值:   %.4f ± %.4f\n", mean(results$r2), sd(results$r2)))

# 保存详细结果
write.csv(results, "elastic_net_glmnet_results.csv", row.names = FALSE)