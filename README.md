# Cardiovascular Disease Risk Prediction Model Based on Multi-dimensional Health Examination Data

本仓库包含一个使用 早期预测心血管疾病风险的项目代码，包括：
- 数据读取与训练 / 验证（`src/main.py`）
- 结果恢复与分析（`src/recover_and_plot_part1.py`）
- 混淆矩阵等可视化（`src/plot_confusion_matrices.py`）

所有实验可以通过一条命令完整复现。

---

## 1. Quick Start（快速上手）

```bash
# 1. 克隆仓库
git clone https://github.com/weixuang/cardiovascular-disease-risk-prediction.git
cd cardiovascular-disease-risk-prediction

# 2. 创建并激活虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 一键运行完整实验流程（训练 + 结果恢复 + 可视化）
./run.sh
```