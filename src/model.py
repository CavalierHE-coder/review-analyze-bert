# -*- coding: utf-8 -*-
"""
@File    : model.py
@Author  : CavalierHE
@Date    : 2026/10/20
@Desc    : 模型定义
"""

import torch.nn as nn
# 导入Hugging Face的AutoModel：自动加载预训练模型
from transformers import AutoModel
from config import *

class ReviewAnalyzeBert(nn.Module):
    def __init__(self,freeze_bert=True):
        super().__init__()

        # 加载本地预训练的BERT-base模型
        # BERT-base模型：层数-12，隐藏维度-768，注意力多头-12
        # self.bert.config.hidden_size          # 768
        # self.bert.config.num_hidden_layers    # 12
        # self.bert.config.num_attention_heads  # 12
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_DIR/'bert-base-chinese')

        # 分类器：接收[cls]向量 -> 输出二分类的得分
        # self.classifier输入：(batch_size,hidden_size)
        # self.classifier输出：(batch_size,1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # 是否冻结BERT参数
        '''
            冻结 BERT：把所有参数的 requires_grad 设为 False，反向传播时不计算梯度、优化器不更新它们。
            这样只训练 classifier，速度快、显存省，适合数据量小的情况。如果 freeze_bert=False，则整个
            BERT 一起微调（效果通常更好，但更慢）。
        '''
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    # 前向传播过程
    def forward(self, input_ids, attention_mask):
        # input_ids.shape：(batch_size,seq_len)
        # attention_mask.shape：(batch_size,seq_len)
        # input_ids：分词后的token_id
        # attention_mask：1表示真实token，0表示padding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        '''
            last_hidden_state[ :,   0,   : ]
                                │    │    └─ 向量的768个数：全要
                                │    └────── 第几个token：只要第0个，[CLS]-相当于整句话的摘要
                                └─────────── 第几句话：全要
        '''
        cls_output = outputs.last_hidden_state[:,0,:]
        # 通过线性层生成logits
        # logits.shape：(batch_size,1)
        logits = self.classifier(cls_output)
        # 返回形状：(batch_size,)
        return logits.squeeze(-1)