import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures, FunctionTransformer, OneHotEncoder
import matplotlib

# ---------------- 设置字体 (防止中文乱码) ----------------
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False 

# ---------------- 基础配置 ----------------
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 复用你之前的核心逻辑 (确保模型一致性)
# ==========================================

def load_data(exp_id: int):
    # 只加载 Fold 0 作为代表
    train_path = os.path.join(DATA_DIR, f"cardio_exp{exp_id}_train.csv")
    test_path  = os.path.join(DATA_DIR, f"cardio_exp{exp_id}_test.csv")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df

def prepare_features(df: pd.DataFrame):
    feature_cols = [
        "age_years", "bmi", "ap_hi", "ap_lo",
        "gender", "cholesterol", "gluc",
        "smoke", "alco", "active",
    ]
    X = df[feature_cols].copy()
    y = df["cardio"].astype(int)
    return X, y

def build_feature_engineering_pipeline():
    numeric_features = ["age_years", "bmi", "ap_hi", "ap_lo"]
    categorical_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

    spline_transformer = SplineTransformer(degree=3, n_knots=10, include_bias=False)
    poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

    def add_manual_features(X):
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

def build_fixed_model(model_name: str):
    pre = build_feature_engineering_pipeline()
    
    if model_name == "logreg":
        return Pipeline([("preprocess", pre), ("clf", LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=42))])
    if model_name == "hgbt":
        return Pipeline([("preprocess", pre), ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, random_state=42))])
    if model_name == "mlp":
        return Pipeline([("preprocess", pre), ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))])
    if model_name == "xgb":
        return Pipeline([("preprocess", pre), ("clf", xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42, n_jobs=-1))])
    
    raise ValueError(f"Unknown model: {model_name}")

# ==========================================
# 新增：专门绘制混淆矩阵的函数
# ==========================================

def plot_confusion_matrix_custom(model, X_test, y_test, model_name):
    """
    绘制美观的混淆矩阵，并打印详细的误诊/漏诊数据
    """
    # 1. 预测
    y_pred = model.predict(X_test)
    
    # 2. 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    # cm[0,0]: TN (真健康)
    # cm[0,1]: FP (误诊 - 假阳性)
    # cm[1,0]: FN (漏诊 - 假阴性)
    # cm[1,1]: TP (确诊 - 真阳性)
    tn, fp, fn, tp = cm.ravel()
    
    # 3. 计算关键指标 (用于报告文本)
    total = np.sum(cm)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    miss_rate = fn / (fn + tp) # 漏诊率
    false_alarm_rate = fp / (fp + tn) # 误报率

    print(f"\n>>> 模型: {model_name} 详细诊断报告")
    print(f"  总样本数: {total}")
    print(f"  真阳性 (TP - 成功确诊): {tp}")
    print(f"  真阴性 (TN - 排除健康): {tn}")
    print(f"  假阳性 (FP - 误诊): {fp} (误报率: {false_alarm_rate:.2%})")
    print(f"  假阴性 (FN - 漏诊): {fn} (漏诊率: {miss_rate:.2%}) <-- 关注这个!")
    print(f"  Recall (召回率): {recall:.4f}")
    print(f"  Precision (精确率): {precision:.4f}")

    # 4. 绘图
    plt.figure(figsize=(7, 6))
    
    # 定义标签
    labels = ['Healthy (0)', 'Disease (1)']
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=15, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # 添加额外的文字说明（可选，显得更专业）
    plt.text(0.5, -0.1, f"Accuracy: {accuracy:.3f} | Recall: {recall:.3f}", 
             ha="center", va="center", transform=plt.gca().transAxes, fontsize=11)

    plt.tight_layout()
    
    # 保存
    filename = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix_exp0_test.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  [图表已保存]: {filename}")

# ==========================================
# 主程序
# ==========================================

def main():
    print("正在加载第 0 折数据 (Fold 0) 作为测试代表...")
    # 加载数据
    train_df, test_df = load_data(0) # 只跑第 0 折
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)
    
    # 定义要运行的模型
    model_list = ["logreg", "xgb", "hgbt", "mlp"]
    
    for model_name in model_list:
        print(f"\n正在训练模型: {model_name} ...")
        
        # 构建并训练
        clf = build_fixed_model(model_name)
        clf.fit(X_train, y_train)
        
        # 绘制混淆矩阵
        plot_confusion_matrix_custom(clf, X_test, y_test, model_name)

    print("\n所有混淆矩阵已生成完毕！请查看 output/ 文件夹。")

if __name__ == "__main__":
    main()