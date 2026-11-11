krr_train_predict <- function(X_train, y_train, X_test, sigma, lambda){
  rbf_kernel <- function(A, B, s){
    sqA <- rowSums(A^2)
    sqB <- rowSums(B^2)
    cross <- tcrossprod(A, B)
    dist_sq <- outer(sqA, sqB, "+") - 2 * cross
    # 修正：sigma直接乘平方距离，和你的 σ=1/d_med 定义匹配
    K <- exp( -s * dist_sq )
    return(K)
  }
  
  K_train <- rbf_kernel(X_train, X_train, sigma)
  n <- nrow(K_train)
  # 增加1e-8对角扰动，防止矩阵奇异
  alpha <- solve(K_train + diag(lambda + 1e-8, n)) %*% y_train
  
  K_test <- rbf_kernel(X_test, X_train, sigma)
  yhat <- as.vector(K_test %*% alpha)
  
  # 兜底：预测值出现NA/Inf时替换为训练集均值
  yhat[is.na(yhat) | is.infinite(yhat)] <- mean(y_train)
  return(yhat)
}
# ==================== 2. 读取数据 ====================
geno <- as.matrix(read.csv("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/geno_for_R.csv", header=FALSE))
pheno <- read.table("C:/Users/19947/Desktop/liqiang/MASPOT_GWAS/data/maspot_qc_pheno.txt", header=FALSE)
# =================================================================
y <- as.numeric(pheno[, ncol(pheno)])
X <- geno
# ---------- 3. 评估参数 ----------
seeds <- c(42,123, 2024, 66, 999)
n_folds <- 5
records <- list()

# 仅 lambda 需要 CV 选择，sigma 将由各折数据自适应固定
lambda_candidates <- c(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100)

# ---------- 4. 核心评估循环 ----------
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
        
        # ----- 特征标准化 -----
        train_mean <- apply(X_train, 2, mean)
        train_sd   <- apply(X_train, 2, sd)
        train_sd[train_sd == 0] <- 1
        X_train_scaled <- scale(X_train, center = train_mean, scale = train_sd)
        X_test_scaled  <- scale(X_test,  center = train_mean, scale = train_sd)
        
        # ----- 固定 sigma：基于训练集距离中位数 -----
        sq_norm <- rowSums(X_train_scaled^2)
        dist_sq <- outer(sq_norm, sq_norm, "+") - 2 * tcrossprod(X_train_scaled)
        d_med <- median(dist_sq[lower.tri(dist_sq)])
        sigma_fixed <- 1 / d_med
        cat(sprintf("种子 %d | 第 %d 折 | 中位数距离^2: %.2f | 固定 sigma: %.6f\n",
                    seed, fold, d_med, sigma_fixed))
        
        # ----- 内层 5 折交叉验证仅选择 lambda -----
        set.seed(seed * 100 + fold)
        inner_folds <- sample(rep(1:5, length.out = length(train_idx)))
        
        best_rmse <- Inf
        best_lambda <- NA
        
        for (l in lambda_candidates) {
            fold_rmse <- numeric(5)
            for (i in 1:5) {
                inner_train <- which(inner_folds != i)
                inner_test  <- which(inner_folds == i)
                
                pred_inner <- tryCatch(
                    krr_train_predict(
                        X_train_scaled[inner_train, ], y_train[inner_train],
                        X_train_scaled[inner_test, ], sigma_fixed, l
                    ),
                    error = function(e) rep(NA, length(inner_test))
                )
                if (any(is.na(pred_inner))) {
                    fold_rmse[i] <- Inf
                } else {
                    fold_rmse[i] <- sqrt(mean((y_train[inner_test] - pred_inner)^2))
                }
            }
            mean_rmse <- mean(fold_rmse)
            if (mean_rmse < best_rmse) {
                best_rmse <- mean_rmse
                best_lambda <- l
            }
        }
        
        # ----- 用固定 sigma 和最优 lambda 预测测试集 -----
        y_pred <- krr_train_predict(
            X_train_scaled, y_train, X_test_scaled,
            sigma_fixed, best_lambda
        )
        
        rmse_test <- sqrt(mean((y_test - y_pred)^2))
        cor_val <- cor(y_test, y_pred)
        r2 <- ifelse(is.na(cor_val), 0, cor_val^2) # 兜底：相关系数NA时R²记为0
        
        records[[length(records) + 1]] <- data.frame(
            seed = seed, fold = fold,
            rmse = rmse_test, r2 = r2,
            best_sigma = sigma_fixed,
            best_lambda = best_lambda
        )
        cat(sprintf("种子 %d | 第 %d 折 | RMSE: %.4f | R²: %.4f | 最优 lambda: %.3f\n",
                    seed, fold, rmse_test, r2, best_lambda))
    }
}
