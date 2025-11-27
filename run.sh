#!/bin/bash

# 确保输出目录存在
mkdir -p output

# Step 1: 运行训练和验证（5 折交叉验证）
echo "正在运行 5 折交叉验证..."
python src/main.py

# Step 2: 结果恢复与分析（如需要生成图表等）
echo "正在恢复结果并生成图像..."
python src/recover_and_plot_part1.py

# Step 3: 生成混淆矩阵等可视化结果
echo "正在生成混淆矩阵和其他可视化结果..."
python src/plot_confusion_matrices.py

echo "所有步骤已完成，结果保存在 output/ 文件夹中。"