import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---- 0. 路径常量（请改成本机实际数据路径）-----------------------------------
DATA_DIR = r"C:\Users\liyili\Desktop\liqiang\MASPOT_GWAS\data"
GENO_NPY = DATA_DIR + r"\maspot_qc_imputed.npy"
PHENO_TXT = DATA_DIR + r"\maspot_qc_pheno.txt"

# ---- 1. 全局参数（对齐论文 4.3.4）------------------------------------------
SEEDS = [42, 66, 123, 999, 2024]
N_FOLDS = 5

POPULATION_SIZE = 50      # 种群规模
MAX_ITERATION = 50         # 最大迭代次数
EARLY_STOP_ROUND = 10     # 连续无改进早停轮数
DISCOVERER_RATIO = 0.20   # 发现者占比
FOLLOWER_RATIO = 0.80     # 跟随者占比
VIGILANTE_RATIO = 0.10    # 警戒者占比 (论文区间 10%-20%)
N_ESTIMATORS = 120

# 树模型参数边界
PARAM_BOUNDS_TREE = {
    "max_depth": (3, 6),
    "learning_rate": (0.05, 0.15),
    "subsample": (0.8, 0.95),
    "colsample_bytree": (0.8, 0.95),
    "reg_lambda": (0.1, 1.0),
}

# 线性模型参数边界
PARAM_BOUNDS_LINEAR = {
    "learning_rate": (0.05, 0.15),
    "reg_lambda": (0.1, 0.8),
    "reg_alpha": (0.01, 0.3),
}


# ---- 2. 通用工具函数 --------------------------------------------------------
def r2_pearson(y_true, y_pred):
    """R2 = 平方 Pearson 相关系数（与论文公式 4 口径一致）。"""
    if np.std(y_pred) == 0:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2


def bound_params_tree(individual):
    individual[0] = np.clip(individual[0], 3, 6)
    individual[1] = np.clip(individual[1], 0.05, 0.15)
    individual[2] = np.clip(individual[2], 0.8, 0.95)
    individual[3] = np.clip(individual[3], 0.8, 0.95)
    individual[4] = np.clip(individual[4], 0.1, 1.0)
    return individual


def bound_params_linear(individual):
    individual[0] = np.clip(individual[0], 0.05, 0.15)
    individual[1] = np.clip(individual[1], 0.1, 0.8)
    individual[2] = np.clip(individual[2], 0.01, 0.3)
    return individual


def cauchy_gaussian_mutation(individual, best_individual, iter_cur, iter_max):
    alpha = 0.8 - 0.6 * (iter_cur / iter_max)
    cauchy = np.random.standard_cauchy(size=individual.shape)
    gauss = np.random.normal(size=individual.shape)
    mutation = alpha * cauchy + (1 - alpha) * gauss
    return individual + mutation * (best_individual - individual)


# ---- 3. 数据加载（全标记，不做 SNP 抽样）-----------------------------------
def load_full_marker_data():
    print("\n" + "=" * 60)
    print("【步骤1】加载全标记基因型与表型")
    print("=" * 60)
    X = np.load(GENO_NPY)
    pheno = pd.read_csv(PHENO_TXT, header=None, sep="\t")
    y = pheno.iloc[:, -1].values
    print(f"样本数: {X.shape[0]} | SNP 数: {X.shape[1]}")
    return X, y


# ---- 4. 适应度评估 ----------------------------------------------------------
def _fit_predict(params, X_tr, y_tr, X_te, y_te, booster):
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        booster=booster,
        early_stopping_rounds=10,
        n_jobs=8,
        random_state=SEEDS[0],
        verbosity=0,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model.predict(X_te)


def fitness_tree(individual, X, y):
    params = {
        "max_depth": int(individual[0]),
        "learning_rate": individual[1],
        "subsample": individual[2],
        "colsample_bytree": individual[3],
        "reg_lambda": individual[4],
    }
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
    y_true, y_pred = [], []
    for tr, te in kf.split(X):
        pred = _fit_predict(params, X[tr], y[tr], X[te], y[te], "gbtree")
        y_pred.extend(pred)
        y_true.extend(y[te])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (r2_pearson(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred))


def fitness_linear(individual, X, y):
    params = {
        "learning_rate": individual[0],
        "reg_lambda": individual[1],
        "reg_alpha": individual[2],
    }
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
    y_true, y_pred = [], []
    for tr, te in kf.split(X):
        pred = _fit_predict(params, X[tr], y[tr], X[te], y[te], "gblinear")
        y_pred.extend(pred)
        y_true.extend(y[te])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return (r2_pearson(y_true, y_pred),
            np.sqrt(mean_squared_error(y_true, y_pred)),
            mean_absolute_error(y_true, y_pred))


# ---- 5. ISSA 优化 -----------------------------------------------------------
def ssa_optimize(fitness_fn, bound_fn, n_params, X, y):
    population = np.random.rand(POPULATION_SIZE, n_params)
    population = np.array([bound_fn(ind) for ind in population])

    fitness_arr = np.array([fitness_fn(ind, X, y)[0] for ind in population])
    best_idx = int(np.argmax(fitness_arr))
    best_fit = fitness_arr[best_idx]
    best_ind = population[best_idx].copy()
    stop_count = 0

    for it in range(MAX_ITERATION):
        # 排序
        order = np.argsort(-fitness_arr)
        population, fitness_arr = population[order], fitness_arr[order]

        # 发现者: 前 20% 全局搜索
        n_discover = int(POPULATION_SIZE * DISCOVERER_RATIO)
        for i in range(n_discover):
            population[i] = population[i] * np.exp(-it / MAX_ITERATION)
            population[i] = bound_fn(population[i])

        # 跟随者: 后 80% 局部跟随
        for i in range(n_discover, POPULATION_SIZE):
            step = np.random.standard_cauchy(size=n_params)
            population[i] = best_ind + step * np.abs(population[i] - best_ind)
            population[i] = bound_fn(population[i])

        # 警戒者: 10%-20% 个体柯西-高斯变异
        n_vigil = max(1, int(POPULATION_SIZE * VIGILANTE_RATIO))
        for i in range(n_vigil):
            population[i], _ = cauchy_gaussian_mutation(
                population[i], best_ind, it, MAX_ITERATION)
            population[i] = bound_fn(population[i])

        # 适应度更新
        new_fit = []
        for ind in population:
            new_fit.append(fitness_fn(ind, X, y)[0])
        new_fit = np.array(new_fit)
        mask = new_fit > fitness_arr
        fitness_arr[mask] = new_fit[mask]

        current_best = np.max(fitness_arr)
        if current_best > best_fit:
            best_fit = current_best
            best_ind = population[int(np.argmax(fitness_arr))].copy()
            stop_count = 0
        else:
            stop_count += 1

        if stop_count >= EARLY_STOP_ROUND:
            print(f"  连续 {EARLY_STOP_ROUND} 轮无改进，早停于第 {it + 1} 轮")
            break

    return best_fit, best_ind


# ---- 6. 外层: 5 种子 × 5 折 重复交叉验证 -----------------------------------
def evaluate(X, y):
    records = []

    for seed in SEEDS:
        np.random.seed(seed)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]

            # 6.1 默认参数 XGB-Tree 基线
            base_model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, booster="gbtree",
                n_jobs=8, random_state=seed, verbosity=0)
            base_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            base_pred = base_model.predict(X_te)
            records.append({
                "seed": seed, "fold": fold, "group": "XGB-Tree-default",
                "r2": r2_pearson(y_te, base_pred),
                "rmse": np.sqrt(mean_squared_error(y_te, base_pred)),
                "mae": mean_absolute_error(y_te, base_pred),
            })

            # 6.2 ISSA 优化 XGB-Tree
            _, best_tree = ssa_optimize(
                fitness_tree, bound_params_tree,
                len(PARAM_BOUNDS_TREE), X_tr, y_tr)
            tree_params = {
                "max_depth": int(best_tree[0]),
                "learning_rate": best_tree[1],
                "subsample": best_tree[2],
                "colsample_bytree": best_tree[3],
                "reg_lambda": best_tree[4],
            }
            tree_model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, booster="gbtree",
                early_stopping_rounds=10, n_jobs=8,
                random_state=seed, verbosity=0, **tree_params)
            tree_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            tree_pred = tree_model.predict(X_te)
            records.append({
                "seed": seed, "fold": fold, "group": "XGB-Tree-ISSA",
                "r2": r2_pearson(y_te, tree_pred),
                "rmse": np.sqrt(mean_squared_error(y_te, tree_pred)),
                "mae": mean_absolute_error(y_te, tree_pred),
            })

            # 6.3 ISSA 优化 XGB-Linear
            _, best_lin = ssa_optimize(
                fitness_linear, bound_params_linear,
                len(PARAM_BOUNDS_LINEAR), X_tr, y_tr)
            lin_params = {
                "learning_rate": best_lin[0],
                "reg_lambda": best_lin[1],
                "reg_alpha": best_lin[2],
            }
            lin_model = xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS, booster="gblinear",
                early_stopping_rounds=10, n_jobs=8,
                random_state=seed, verbosity=0, **lin_params)
            lin_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            lin_pred = lin_model.predict(X_te)
            records.append({
                "seed": seed, "fold": fold, "group": "XGB-Linear-ISSA",
                "r2": r2_pearson(y_te, lin_pred),
                "rmse": np.sqrt(mean_squared_error(y_te, lin_pred)),
                "mae": mean_absolute_error(y_te, lin_pred),
            })

            print(f"种子 {seed} | 折 {fold} 完成")

    return pd.DataFrame(records)


# ---- 7. 主程序 --------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_full_marker_data()
    results = evaluate(X, y)

    print("\n" + "=" * 60)
    print("各组 25 次重复交叉验证汇总")
    print("=" * 60)
    for group in results["group"].unique():
        sub = results[results["group"] == group]
        print(f"{group:18s} R2={sub['r2'].mean():.4f}±{sub['r2'].std():.4f} | "
              f"RMSE={sub['rmse'].mean():.4f}±{sub['rmse'].std():.4f} | "
              f"MAE={sub['mae'].mean():.4f}±{sub['mae'].std():.4f}")

    results.to_csv("xgboost_ssa_results.csv", index=False)
