# -*- coding: utf-8 -*-
"""
@File    : config.py
@Author  : CavalierHE
@Date    : 2026/10/19
@Desc    : 配置文件
"""

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 路径设置
MODEL_DIR = BASE_DIR / 'models'  # 模型参数保存路径
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'  # 已处理数据存放路径(如token序列)
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'  # 原始CSV或文本数据路径
LOGS_DIR = BASE_DIR / 'logs'  # TensorBoard日志保存路径
PRE_TRAINED_DIR = BASE_DIR / 'pretrained'  # 本地预训练模型存放路径

# 训练超参数
SEQ_LEN = 64  # 最大序列长度
BATCH_SIZE = 64  # 批处理大小
LEARNING_RATE = 1e-5  # 学习率
EPOCHS = 20  # 训练轮数