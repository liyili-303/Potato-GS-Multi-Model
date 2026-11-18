DATA_DIR  <- "C:/Users/liyili/Desktop/liqiang/MASPOT_GWAS/data"
GENO_FILE <- file.path(DATA_DIR, "geno_for_R.csv")
PHENO_FILE <- file.path(DATA_DIR, "maspot_qc_pheno.txt")

# ---- 1. 读取数据 ------------------------------------------------------------
geno <- as.matrix(read.csv(GENO_FILE, header = FALSE))
pheno <- read.table(PHENO_FILE, header = FALSE)

y <- as.numeric(pheno[, ncol(pheno)])
X <- geno

# ---- 2. 评估设置 ------------------------------------------------------------
seeds <- c(42, 66, 123, 999, 2024)
n_folds <- 5

lambda_candidates <- c(0.001, 0.003, 0.005, 0.008, 0.01,
                       0.02, 0.03, 0.05, 0.07, 0.1)
sigma_grid <- c(0.000005, 0.000010, 0.000014, 0.000020, 0.000040)

# RBF 核 + 求解
krr_train_predict <- function(X_train, y_train, X_test, sigma, lambda) {
  rbf_kernel <- function(A, B, s) {
    sqA <- rowSums(A^2)
    sqB <- rowSums(B^2)
    cross <- tcrossprod(A, B)
    dist_sq <- outer(sqA, sqB, "+") - 2 * cross
    exp(-s * dist_sq)
  }
  K_train <- rbf_kernel(X_train, X_train, sigma)
  n <- nrow(K_train)
  alpha <- solve(K_train + diag(lambda + 1e-8, n)) %*% y_train
  K_test <- rbf_kernel(X_test, X_train, sigma)
  yhat <- as.vector(K_test %*% alpha)
  yhat[is.na(yhat) | is.infinite(yhat)] <- mean(y_train)
  yhat
}

r2_score <- function(y_obs, y_pred) {
  cv <- cor(y_obs, y_pred)
  ifelse(is.na(cv), 0, cv^2)
}

records_fixed <- list()   # 主实验: 固定 sigma 启发式
records_grid  <- list()   # 对照: sigma × λ 联合网格
records_def   <- list()   # 默认基线

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

    # 训练集标准化, 测试集复用训练集 center/scale
    train_mean <- apply(X_train, 2, mean)
    train_sd   <- apply(X_train, 2, sd)
    train_sd[train_sd == 0] <- 1
    X_train_scaled <- scale(X_train, center = train_mean, scale = train_sd)
    X_test_scaled  <- scale(X_test,  center = train_mean, scale = train_sd)

    # 训练集中位数距离 -> 启发式 sigma
    sq_norm <- rowSums(X_train_scaled^2)
    dist_sq <- outer(sq_norm, sq_norm, "+") - 2 * tcrossprod(X_train_scaled)
    d_med   <- median(dist_sq[lower.tri(dist_sq)])
    sigma_fixed <- 1 / d_med

    # 内层 5 折索引
    set.seed(seed * 100 + fold)
    inner_folds <- sample(rep(1:n_folds, length.out = length(train_idx)))

    inner_cv_lambda <- function(lambda) {
      inner_rmse <- numeric(n_folds)
      for (i in 1:n_folds) {
        itr <- which(inner_folds != i)
        ite <- which(inner_folds == i)
        pred <- tryCatch(
          krr_train_predict(X_train_scaled[itr, ], y_train[itr],
                           X_train_scaled[ite, ], sigma_fixed, lambda),
          error = function(e) rep(NA, length(ite))
        )
        inner_rmse[i] <- if (any(is.na(pred))) Inf else
          sqrt(mean((y_train[ite] - pred)^2))
      }
      mean(inner_rmse)
    }

    # ----- 3.1 主实验: 固定 sigma, 内层选 λ -----
    best_lambda <- NA
    best_inner_rmse <- Inf
    for (l in lambda_candidates) {
      m <- inner_cv_lambda(l)
      if (m < best_inner_rmse) {
        best_inner_rmse <- m
        best_lambda <- l
      }
    }
    y_pred_fixed <- krr_train_predict(X_train_scaled, y_train,
                                      X_test_scaled, sigma_fixed, best_lambda)
    records_fixed[[length(records_fixed) + 1]] <- data.frame(
      seed = seed, fold = fold,
      rmse = sqrt(mean((y_test - y_pred_fixed)^2)),
      r2   = r2_score(y_test, y_pred_fixed),
      best_sigma = sigma_fixed, best_lambda = best_lambda
    )

    # ----- 3.2 网格对照: sigma × λ 联合内层 5 折 CV -----
    best_grid_rmse <- Inf
    best_grid_sigma <- NA
    best_grid_lambda <- NA
    for (s in sigma_grid) {
      for (l in lambda_candidates) {
        inner_rmse <- numeric(n_folds)
        for (i in 1:n_folds) {
          itr <- which(inner_folds != i)
          ite <- which(inner_folds == i)
          pred <- tryCatch(
            krr_train_predict(X_train_scaled[itr, ], y_train[itr],
                              X_train_scaled[ite, ], s, l),
            error = function(e) rep(NA, length(ite))
          )
          inner_rmse[i] <- if (any(is.na(pred))) Inf else
            sqrt(mean((y_train[ite] - pred)^2))
        }
        m <- mean(inner_rmse)
        if (m < best_grid_rmse) {
          best_grid_rmse   <- m
          best_grid_sigma  <- s
          best_grid_lambda  <- l
        }
      }
    }
    y_pred_grid <- krr_train_predict(X_train_scaled, y_train,
                                     X_test_scaled, best_grid_sigma, best_grid_lambda)
    records_grid[[length(records_grid) + 1]] <- data.frame(
      seed = seed, fold = fold,
      rmse = sqrt(mean((y_test - y_pred_grid)^2)),
      r2   = r2_score(y_test, y_pred_grid),
      best_sigma = best_grid_sigma, best_lambda = best_grid_lambda
    )

    # ----- 3.3 默认基线: sigma = 1/n_features, λ = 1.0 -----
    sigma_def <- 1 / ncol(X_train_scaled)
    y_pred_def <- krr_train_predict(X_train_scaled, y_train,
                                    X_test_scaled, sigma_def, 1.0)
    records_def[[length(records_def) + 1]] <- data.frame(
      seed = seed, fold = fold,
      rmse = sqrt(mean((y_test - y_pred_def)^2)),
      r2   = r2_score(y_test, y_pred_def),
      best_sigma = sigma_def, best_lambda = 1.0
    )

    cat(sprintf("种子 %d | 第 %d 折 | 固定σ R2=%.4f | 网格 R2=%.4f | 默认 R2=%.4f\n",
                seed, fold,
                r2_score(y_test, y_pred_fixed),
                r2_score(y_test, y_pred_grid),
                r2_score(y_test, y_pred_def)))
  }
}

# ---- 4. 汇总 ----------------------------------------------------------------
results_fixed <- do.call(rbind, records_fixed)
results_grid  <- do.call(rbind, records_grid)
results_def   <- do.call(rbind, records_def)

cat("\n==================== KRR 主实验(固定σ) ====================\n")
print(results_fixed)
cat(sprintf("Mean R2 = %.4f ± %.4f | Mean RMSE = %.4f ± %.4f\n",
            mean(results_fixed$r2), sd(results_fixed$r2),
            mean(results_fixed$rmse), sd(results_fixed$rmse)))

cat("\n==================== KRR 网格对照 ====================\n")
print(results_grid)
cat(sprintf("Mean R2 = %.4f ± %.4f | Mean RMSE = %.4f ± %.4f\n",
            mean(results_grid$r2), sd(results_grid$r2),
            mean(results_grid$rmse), sd(results_grid$rmse)))

cat("\n==================== KRR 默认基线 ====================\n")
print(results_def)
cat(sprintf("Mean R2 = %.4f ± %.4f | Mean RMSE = %.4f ± %.4f\n",
            mean(results_def$r2), sd(results_def$r2),
            mean(results_def$rmse), sd(results_def$rmse)))

write.csv(results_fixed, "krr_fixed_sigma_results.csv", row.names = FALSE)
write.csv(results_grid,  "krr_grid_results.csv", row.names = FALSE)
write.csv(results_def,   "krr_default_results.csv", row.names = FALSE)
