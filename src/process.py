# -*- coding: utf-8 -*-
"""
@File    : process.py
@Author  : CavalierHE
@Date    : 2026/10/19
@Desc    : 数据预处理
"""

from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from config import *

def process_data():
    # 加载原始CSV数据
    dataset = load_dataset('csv',data_files=str(RAW_DATA_DIR/'online_shopping_10_cats.csv'))['train']  # 返回字典，key为train

    # 过滤空评论和非二分类标签
    dataset = dataset.filter(
        lambda x:x['review'] is not None
            and x['review'].strip() != ''
            and x['label'] in [0,1]
    )

    # 划分训练集和测试集
    dataset = dataset.cast_column("label", ClassLabel(names=["neg","pos"]))
    # 分层抽样，切出来的训练集、测试集里面，0和1的比例依然维持80%:20%
    dataset_dict = dataset.train_test_split(test_size=0.2,seed=42,stratify_by_column='label')
    print("数据划分完成")

    # 加载分词器
    '''
    自动在'bert-base-chinese'目录搜索分词器相关文件：
        1. 先读 `tokenizer_config.json` -> 获取分词器类型、特殊符号
        2. 读取 `vocab.txt` -> 建立文字 <-> token id 的映射
        3. 读取 `tokenizer.json` -> 分词算法
    '''
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR / 'bert-base-chinese')

    # 编码函数
    def tokenize(example):
        encoded = tokenizer(
            example['review'],  # 待处理文本：评论内容
            max_length=SEQ_LEN,  # 最大序列长度（SEQ_LEN是config里定义的常数，64）
            truncation=True,  # 截断开关：文本超过max_length，直接砍掉后面多余的token
            padding='max_length'  # 填充：文本不足max_length，在末尾补[PAD]占位符，统一长度
        )
        return {
            'input_ids': encoded['input_ids'],  # input_ids：token编号列表，汉字/符号 -> 在词表中对应的数字ID
            'attention_mask': encoded['attention_mask']  # 注意力掩码，告诉BERT哪些token是真实文本、哪些是补的PAD填充位
        }

    # 对训练和测试集分布编码
    dataset_dict = dataset_dict.map(tokenize, batched=True)
    print("分词完成")

    # 删除字段
    dataset_dict = dataset_dict.remove_columns(['review', 'cat'])

    # 保存处理结果
    dataset_dict['train'].save_to_disk(str(PROCESSED_DATA_DIR/'train'))
    dataset_dict['test'].save_to_disk(str(PROCESSED_DATA_DIR/'test'))
    print("保存完成")

if __name__ == '__main__':
    process_data()