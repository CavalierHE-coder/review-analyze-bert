# -*- coding: utf-8 -*-
"""
@File    : predict.py
@Author  : CavalierHE
@Date    : 2026/10/20
@Desc    : 模型预测
"""

import torch
from transformers import AutoTokenizer
from config import *
from model import ReviewAnalyzeBert

# 对一个batch的输入进行预测，返回sigmoid概率
def predict_batch(input_ids,attention_mask,model):
    model.eval()

    # input_ids.shape: (batch_size,seq_len)
    # attention_mask.shape: (batch_size,seq_len)

    with torch.no_grad():
        # logits.shape: (batch_size,)
        logits = model(input_ids=input_ids,attention_mask=attention_mask)
        # 概率值∈[0,1]，表示为正面情感的置信度
        probs = torch.sigmoid(logits)
        return probs.tolist()  # 返回Python列表

def predict_text(user_input,model,tokenizer,device):
    # 文本编码为张量形式(长度固定)
    encoded = tokenizer(
        user_input,
        max_length=SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)  # input_ids.shape: (1,seq_len)
    attention_mask = encoded['attention_mask'].to(device)  # attention_mask.shape: (1,seq_len)

    # 模型预测
    prob = predict_batch(input_ids,attention_mask,model)[0]
    return prob

# 交互式预测主程序
def run_predict():
    # 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR/'bert-base-chinese')
    model = ReviewAnalyzeBert().to(device)
    model.load_state_dict(torch.load(MODEL_DIR/'model.pt',map_location=device))

    # 命令行交互循环
    print('请输入评价（输入q或者quit退出）：')
    while True:
        user_input = input('> ').strip()
        # 输入为空或退出
        if user_input.lower() in {'q', 'quit', 'exit'}:
            print('感谢使用，再见！')
            break
        if not user_input:
            print('输入不能为空，请重新输入！')
            continue

        result = predict_text(user_input,model,tokenizer,device)

        # 显示结果
        if result > 0.5:
            print(f"正面评价（置信度：{result:.2f}）")
        else:
            print(f"负面评价（置信度：{1 - result:.2f}）")

if __name__ == '__main__':
    run_predict()