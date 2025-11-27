import os 
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]      # 或 ["Microsoft YaHei"]，按你系统字体来
matplotlib.rcParams["axes.unicode_minus"] = False   
# ---------------- 基本路径 ----------------

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 数据加载 & 特征准备 ----------------

def load_data(exp_id: int):
    train_path = os.path.join(DATA_DIR, f"cardio_exp{exp_id}_train.csv")
    val_path   = os.path.join(DATA_DIR, f"cardio_exp{exp_id}_val.csv")
    test_path  = os.path.join(DATA_DIR, f"cardio_exp{exp_id}_test.csv")

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "age_years", "bmi", "ap_hi", "ap_lo",
        "gender", "cholesterol", "gluc",
        "smoke", "alco", "active",
    ]
    for col in feature_cols + ["cardio"]:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}，请确认是否已经运行 clean_cardio 并保存。")

    X = df[feature_cols].copy()
    y = df["cardio"].astype(int)
    return X, y

# ---------------- 特征工程流水线 ----------------

def build_feature_engineering_pipeline():
    """
    延续你现在的思路：Spline + Poly + OneHot。
    手动特征这版为了稳，不在 ColumnTransformer 里再折腾高维特征名，
    统一用 permutation_importance 在原始 10 个特征上算重要性。
    """
    numeric_features = ["age_years", "bmi", "ap_hi", "ap_lo"]
    categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

    spline_transformer = SplineTransformer(
        degree=3,
        n_knots=10,
        include_bias=False
    )
    poly_features = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )

    def add_manual_features(X):
        # 这里 X 只有 numeric_features，不再尝试构造依赖 smoke/alco 的手工特征，避免维度乱掉
        # 你要更激进的手工特征可以单独在 prepare_features 里做（原始 10 特征层面）
        return X

    preprocess = ColumnTransformer(
        transformers=[
            ("spline", spline_transformer, numeric_features),
            ("poly",   poly_features,      numeric_features),
            ("manual", FunctionTransformer(add_manual_features, validate=False), numeric_features),
            ("cat",    OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocess

# ---------------- 模型构建 ----------------

def build_fixed_model(model_name: str):
    """
    所有模型统一成：Pipeline(预处理 + 模型)，
    这样上层代码只关心 clf.predict_proba / clf.score。
    """
    pre = build_feature_engineering_pipeline()

    if model_name == "logreg":
        clf = Pipeline([
            ("preprocess", pre),
            ("clf", LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=2000,
                random_state=42,
            )),
        ])
        return clf

    if model_name == "hgbt":
        clf = Pipeline([
            ("preprocess", pre),
            ("clf", HistGradientBoostingClassifier(
                loss="log_loss",
                max_depth=5,
                learning_rate=0.05,
                max_leaf_nodes=30,
                min_samples_leaf=50,
                max_iter=200,
                random_state=42,
            )),
        ])
        return clf

    if model_name == "mlp":
        clf = Pipeline([
            ("preprocess", pre),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.0001,
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                random_state=42,
            )),
        ])
        return clf

    if model_name == "xgb":
        clf = Pipeline([
            ("preprocess", pre),
            ("clf", xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
            )),
        ])
        return clf

    raise ValueError(f"未知模型类型: {model_name}")

# ---------------- 单次评估 ----------------

def eval_split(clf, X, y):
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "auc_roc": roc_auc_score(y, y_prob),
    }

# ---------------- 数据概况 & 单变量关联 ----------------

def describe_and_univariate_analysis(df_all: pd.DataFrame):
    print("===== 数据概况 =====")
    n = len(df_all)
    prevalence = df_all["cardio"].mean()
    print(f"样本量: {n}")
    print(f"患病率: {prevalence:.3f}")

    num_cols = ["age_years", "bmi", "ap_hi", "ap_lo"]
    corr = df_all[num_cols + ["cardio"]].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("数值特征与心血管疾病相关性矩阵")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "corr_heatmap.png"))
    plt.close()

    # 单变量：按分段看患病率
    for col, bins in [
        ("age_years", [30, 40, 50, 60, 70, 80]),
        ("bmi",       [18.5, 25, 30, 35, 40]),
        ("ap_hi",     [100, 120, 140, 160, 180]),
    ]:
        df = df_all.copy()
        df[f"{col}_bin"] = pd.cut(df[col], bins=bins)
        grp = df.groupby(f"{col}_bin", observed=False)["cardio"].mean()
        plt.figure()
        grp.plot(kind="bar")
        plt.ylabel("患病率")
        plt.title(f"{col} 分段与患病率")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{col}_bin_prevalence.png"))
        plt.close()

# ---------------- 多模型 ROC / 校准 ----------------

def plot_roc_multi(models, X_val, y_val, tag="val"):
    plt.figure()
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc = roc_auc_score(y_val, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC 曲线对比 ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_multi_{tag}.png"))
    plt.close()


def plot_calibration_multi(models, X_val, y_val, tag="val"):
    plt.figure()
    for name, clf in models.items():
        y_prob = clf.predict_proba(X_val)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=10)
        plt.plot(mean_pred, frac_pos, "s-", label=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    plt.xlabel("预测概率均值")
    plt.ylabel("阳性比例")
    plt.title(f"校准曲线对比 ({tag})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"calibration_multi_{tag}.png"))
    plt.close()

# ---------------- permutation 特征重要性 ----------------

def plot_perm_importance(models, X_val, y_val):
    """
    对原始 10 个输入特征做 permutation importance，
    不再去管 OneHot / Spline 之后的高维空间，避免 feature_names 对不上。
    """
    for name, clf in models.items():
        print(f"计算置换特征重要性: {name}")
        r = permutation_importance(
            clf,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importances_mean = r.importances_mean
        indices = np.argsort(importances_mean)[::-1]
        topk = indices[:10]

        plt.figure(figsize=(6, 4))
        sns.barplot(
            x=importances_mean[topk],
            y=X_val.columns[topk]
        )
        plt.xlabel("平均精度下降")
        plt.title(f"{name} 置换特征重要性 (前10)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"perm_importance_{name}.png"))
        plt.close()

# ---------------- PDP 风险因子曲线 ----------------

def plot_pdp_for_models(models, X_val):
    features = ["age_years", "ap_hi", "bmi"]
    for name, clf in models.items():
        try:
            disp = PartialDependenceDisplay.from_estimator(
                clf,
                X_val,
                features=features,
                grid_resolution=40,
            )
            disp.figure_.suptitle(f"{name} 部分依赖曲线")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"pdp_{name}.png"))
            plt.close(disp.figure_)
        except Exception as e:
            print(f"PDP 绘制失败: {name}: {e}")

# ---------------- 总控：多模型 × 5折 ----------------

def run_all_folds(model_names=None):
    if model_names is None:
        model_names = ["logreg", "xgb", "hgbt", "mlp"]

    # metrics[model][split][metric] -> list over folds
    metrics = {m: {"val": {}, "test": {}} for m in model_names}
    for m in model_names:
        for split in ["val", "test"]:
            for k in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                metrics[m][split][k] = []

    rep_models = {}   # 第 0 折训练好的模型，用来做各种可视化
    rep_val = rep_test = None
    all_trainvaltest_0 = None

    for exp_id in range(5):
        print(f"\n=== 运行折 {exp_id} ===")
        train_df, val_df, test_df = load_data(exp_id)
        X_train, y_train = prepare_features(train_df)
        X_val,   y_val   = prepare_features(val_df)
        X_test,  y_test  = prepare_features(test_df)

        if exp_id == 0:
            # 用第 0 折的 train+val+test 做整体数据分析
            all_trainvaltest_0 = pd.concat([train_df, val_df, test_df], axis=0)

        for m in model_names:
            print(f"  训练模型: {m}")
            clf = build_fixed_model(m)
            clf.fit(X_train, y_train)

            val_metrics  = eval_split(clf, X_val,  y_val)
            test_metrics = eval_split(clf, X_test, y_test)

            for k, v in val_metrics.items():
                metrics[m]["val"][k].append(v)
            for k, v in test_metrics.items():
                metrics[m]["test"][k].append(v)

            if exp_id == 0:
                rep_models[m] = clf
                rep_val  = (X_val.copy(),  y_val.copy())
                rep_test = (X_test.copy(), y_test.copy())

    # -------- 结果汇总 --------
    summary = {"summary": {}}
    for m in model_names:
        summary["summary"][m] = {}
        for split in ["val", "test"]:
            summary["summary"][m][split] = {}
            for k, vals in metrics[m][split].items():
                summary["summary"][m][split][k] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                }

    out_path = os.path.join(OUTPUT_DIR, "model_summary_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 各模型 5 折平均结果 =====")
    for m in model_names:
        print(f"\n模型: {m}")
        for split in ["val", "test"]:
            line = [split]
            for k in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                v = summary["summary"][m][split][k]["mean"]
                line.append(f"{k}={v:.3f}")
            print("  " + " | ".join(line))

    # -------- 数据概况 & 单变量关联（用第 0 折的整体数据） --------
    if all_trainvaltest_0 is not None:
        describe_and_univariate_analysis(all_trainvaltest_0)

    # -------- 可视化：ROC / Calibration / Permutation Importance / PDP --------
    if rep_models:
        Xv, yv = rep_val
        # ROC & 校准曲线用第 0 折的 val 作为代表
        plot_roc_multi(rep_models, Xv, yv, tag="val_fold0")
        plot_calibration_multi(rep_models, Xv, yv, tag="val_fold0")
        # permutation importance & PDP
        plot_perm_importance(rep_models, Xv, yv)
        plot_pdp_for_models(rep_models, Xv)


if __name__ == "__main__":
    # 同时跑多个模型，并自动做结果对比与可视化
    run_all_folds(model_names=["logreg", "xgb", "hgbt", "mlp"])






