# -*- coding: utf-8 -*-
"""
@File    : evaluate.py
@Author  : CavalierHE
@Date    : 2026/10/20
@Desc    : 模型评估
"""

import torch
from tqdm import tqdm
from config import *
from dataset import get_dataloader
from model import ReviewAnalyzeBert
from predict import predict_batch

def evaluate_model(dataloader, model, device):
    correct = 0
    total = 0
    # 【新增】混淆矩阵的四个计数(以正面=1为正类)
    tp = 0  # 真实正面,预测正面
    fp = 0  # 真实负面,预测正面(误报)
    fn = 0  # 真实正面,预测负面(漏掉)
    tn = 0  # 真实负面,预测负面

    for batch in tqdm(dataloader, desc="评估"):
        # input_ids.shape：(batch_size, seq_len)
        input_ids = batch['input_ids'].to(device)
        # attention_mask.shape：(batch_size, seq_len)
        attention_mask = batch['attention_mask'].to(device)
        # labels.shape：(batch_size,)
        labels = batch['label'].to(device)

        # 预测每个样本的正面情感概率
        # probs是float列表
        probs = predict_batch(input_ids, attention_mask, model)

        # 将概率转换为预测标签(>= 0.5 为正面)
        preds = [1 if p >= 0.5 else 0 for p in probs]

        # 计算准确数量
        for pred, label in zip(preds, labels):
            label = int(label.item())
            if pred == label:
                correct += 1
            total += 1

            # 【新增】统计混淆矩阵
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

    # 【新增】指标计算放在循环外,评估完全部数据后只输出一次
    # 准确率 = 预测对的 / 总数
    acc = correct / total if total > 0 else 0
    # 精确率 = 预测为正面的里面,真的是正面的比例
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # 召回率 = 真实正面的里面,被找出来的比例
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # F1 = 精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 输出评估结果
    print("========== 评估结果 ==========")
    print(f"样本总数: {total}")
    print(f"混淆矩阵: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"准确率(Accuracy):  {acc:.4f}")
    print(f"精确率(Precision): {precision:.4f}")
    print(f"召回率(Recall):    {recall:.4f}")
    print(f"F1:               {f1:.4f}")
    print("============================")

def run_evaluate():
    # 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = ReviewAnalyzeBert().to(device)
    model.load_state_dict(torch.load(MODEL_DIR/'model.pt', map_location=device))
    # 加载测试数据集
    dataloader = get_dataloader(train=False)
    # 执行评估
    evaluate_model(dataloader, model, device)

if __name__ == '__main__':
    run_evaluate()

'''
========== 评估结果 ==========
样本总数: 12555
混淆矩阵: TP=5958  FP=308  FN=388  TN=5901
准确率(Accuracy):  0.9446
精确率(Precision): 0.9508
召回率(Recall):    0.9389
F1:                0.9448
============================
'''