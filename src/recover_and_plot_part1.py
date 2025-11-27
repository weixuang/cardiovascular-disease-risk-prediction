import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from math import pi

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer, OneHotEncoder, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# 设置英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. 定义模型和数据处理 (复用之前的逻辑)
# ==========================================

def load_data(exp_id):
    train = pd.read_csv(f"{DATA_DIR}/cardio_exp{exp_id}_train.csv")
    test = pd.read_csv(f"{DATA_DIR}/cardio_exp{exp_id}_test.csv")
    return train, test

def prepare_features(df):
    feature_cols = ["age_years", "bmi", "ap_hi", "ap_lo", "gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    X = df[feature_cols].copy()
    y = df["cardio"].astype(int)
    return X, y

def build_model(name):
    numeric_features = ["age_years", "bmi", "ap_hi", "ap_lo"]
    categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    
    # 特征工程
    pre = ColumnTransformer([
        ("spline", SplineTransformer(degree=3, n_knots=10, include_bias=False), numeric_features),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), numeric_features),
        ("manual", FunctionTransformer(lambda x: x, validate=False), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    if name == "logreg":
        return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, solver='liblinear', random_state=42))])
    if name == "hgbt":
        return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42))])
    if name == "mlp":
        return Pipeline([("pre", pre), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))])
    if name == "xgb":
        return Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=-1, random_state=42))])

# ==========================================
# 2. 核心：重新计算 5 折结果并生成 JSON
# ==========================================

def calculate_and_save_json():
    models = ["logreg", "xgb", "hgbt", "mlp"]
    # 存储结构: metrics[model][metric] = [fold0_score, fold1_score, ...]
    metrics_storage = {m: {k: [] for k in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]} for m in models}

    print("🚀 开始重新计算 5 折交叉验证结果 (这可能需要几分钟)...")
    
    for i in range(5):
        print(f"  正在处理第 {i}/4 折...")
        train_df, test_df = load_data(i)
        X_train, y_train = prepare_features(train_df)
        X_test, y_test = prepare_features(test_df)

        for m in models:
            clf = build_model(m)
            clf.fit(X_train, y_train)
            
            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics_storage[m]["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics_storage[m]["precision"].append(precision_score(y_test, y_pred))
            metrics_storage[m]["recall"].append(recall_score(y_test, y_pred))
            metrics_storage[m]["f1_score"].append(f1_score(y_test, y_pred))
            metrics_storage[m]["auc_roc"].append(roc_auc_score(y_test, y_prob))

    # 计算均值和标准差并构建最终 JSON 结构
    final_summary = {"summary": {}}
    for m in models:
        final_summary["summary"][m] = {"test": {}}
        for k in metrics_storage[m]:
            vals = metrics_storage[m][k]
            final_summary["summary"][m]["test"][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals))
            }

    json_path = os.path.join(OUTPUT_DIR, "model_summary_results.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"✅ JSON 文件已恢复: {json_path}")
    return final_summary["summary"]

# ==========================================
# 3. 绘图功能 (直接画第一部分需要的两张图)
# ==========================================

def plot_charts(summary_data):
    print("🎨 正在生成第一部分图表...")
    
    # 准备 DataFrame
    rows = []
    name_map = {"logreg": "LogReg", "xgb": "XGBoost", "hgbt": "HGBT", "mlp": "MLP"}
    
    for m, metrics in summary_data.items():
        test_metrics = metrics["test"]
        for k, v in test_metrics.items():
            rows.append({"Model": name_map.get(m, m), "Metric": k, "Mean": v["mean"]})
    
    df = pd.DataFrame(rows)

    # --- 1. 柱状图 ---
    metric_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1-Score', 'auc_roc': 'AUC-ROC'}
    df['Metric Label'] = df['Metric'].map(metric_map)
    df = df[df['Metric Label'].notna()]

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    chart = sns.barplot(data=df, x='Metric Label', y='Mean', hue='Model', palette='viridis')
    plt.ylim(0.6, 0.85)
    plt.title('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold')
    for c in chart.containers: chart.bar_label(c, fmt='%.3f', padding=3, fontsize=9)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "final_performance_bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. 雷达图 ---
    categories = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels, size=12)
    plt.yticks([0.65, 0.70, 0.75, 0.80], ["0.65", "0.70", "0.75", "0.80"], color="grey", size=10)
    plt.ylim(0.60, 0.83)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # LogReg, XGB, HGBT, MLP
    markers = ['o', 's', '^', 'D']

    for idx, (m, metrics) in enumerate(summary_data.items()):
        values = [metrics["test"][cat]["mean"] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name_map.get(m, m), color=colors[idx], marker=markers[idx])
        ax.fill(angles, values, alpha=0.05, color=colors[idx])

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Comprehensive Model Capabilities', size=20, y=1.1)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_performance_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🎉 图表已生成: final_performance_bar.png 和 final_performance_radar.png")

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    # 1. 计算并生成 JSON
    data = calculate_and_save_json()
    
    # 2. 画图
    plot_charts(data)