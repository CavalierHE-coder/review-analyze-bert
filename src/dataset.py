# -*- coding: utf-8 -*-
"""
@File    : dataset.py
@Author  : CavalierHE
@Date    : 2026/10/20
@Desc    : 自定义数据集
"""

from datasets import load_from_disk
from torch.utils.data import DataLoader
from config import *

def get_dataset(train=True):
    path = PROCESSED_DATA_DIR / ('train' if train else 'test')
    dataset = load_from_disk(str(path))

    # 设置为PyTorch格式，列自动转换为tensor
    dataset.set_format(type='torch',columns=['input_ids','attention_mask','label'])
    '''
        {
        'input_ids':      tensor([101, 2769, 3221, ..., 102, 0, 0]),   # (seq_len,)
        'attention_mask': tensor([1, 1, 1, ..., 1, 0, 0]),             # (seq_len,)
        'label':          tensor(1)                                    # 标量
        }
    '''
    return dataset

def get_dataloader(train=True):
    dataset = get_dataset(train)
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

# 数据加载测试入口
if __name__ == '__main__':
    dataloader = get_dataloader(train=True)
    for batch in dataloader:
        # 字典推导式，遍历batch字典里所有的(key,value)键值对
        print({k:v.shape for k,v in batch.items()})
        break