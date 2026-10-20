# -*- coding: utf-8 -*-
"""
@File    : train.py
@Author  : CavalierHE
@Date    : 2026/10/20
@Desc    : 模型训练
"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import *
from dataset import get_dataloader
from model import ReviewAnalyzeBert

# 【新增】显存优化相关配置
ACCUM_STEPS = 1            # 梯度累积步数:等效 batch = BATCH_SIZE * ACCUM_STEPS
USE_AMP = True             # 混合精度


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    # 【新增】set_to_none=True 比置零更省显存
    # 原因：
    # 1》更快：省掉了一次对所有参数梯度的清零写操作。
    # 2》反向传播时少一次加法：梯度为 None 时,第一次反向直接赋值,而不是"旧值 + 新值"。
    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(tqdm(dataloader, desc="训练")):
        # input_ids.shape：(batch_size,seq_len)
        input_ids = batch['input_ids'].to(device)
        # attention_mask.shape：(batch_size,seq_len)
        attention_mask = batch['attention_mask'].to(device)
        # labels.shape：(batch_size,)
        labels = batch['label'].float().to(device)

        # 模型前向传播
        # outputs.shape：(batch_size,)
        # 【新增】bf16 混合精度前向:激活值以 bf16 存储,显存约减半
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=USE_AMP and device.type == "cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 计算损失
            loss = loss_fn(outputs.float(), labels)

        # 反向传播并更新参数
        # 【新增】梯度累积:损失按累积步数缩放,累积够了再更新
        (loss / ACCUM_STEPS).backward()
        '''
        batch 1: backward → 梯度累加(不更新)
        batch 2: backward → 梯度累加(不更新)
        batch 3: backward → 梯度累加(不更新)
        batch 4: backward → 梯度累加 → 更新参数 → 清空梯度
        batch 5: ...(重复)
        '''
        is_update_step = (
                (batch_index + 1) % ACCUM_STEPS == 0  # 每攒够ACCUM_STEPS个batch更新一次
                or (batch_index + 1) == len(dataloader))  # 最后一个batch无论攒没攒够,都要更新一次,否则尾部的梯度会被浪费
        if is_update_step:
            optimizer.step()
            # 参数更新完，清除历史梯度
            optimizer.zero_grad(set_to_none=True)

        # 统计与显示损失
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# 模型训练主函数
def train():
    # 选择运行设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 加载训练数据集
    dataloader = get_dataloader(train=True)
    print("数据集加载完成")

    # 初始化模型并移动设备
    model = ReviewAnalyzeBert(freeze_bert=False).to(device)

    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, fused=True)

    # 使用带sigmoid的二分类交叉熵损失函数
    loss_function = torch.nn.BCEWithLogitsLoss()

    #初始化TensorBoard写入器
    log_dir = LOGS_DIR/time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir))

    # 多轮训练
    best_loss = float("inf")
    for epoch in range(1,EPOCHS+1):
        print(f"============ Epoch {epoch} ===========")
        avg_loss = train_one_epoch(model, dataloader, optimizer, loss_function, device)
        print(f"训练集loss: {avg_loss:.4f}")

        # 写入TensorBoard日志
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # 保存训练好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),MODEL_DIR/'model.pt')

        # 【新增】每轮结束释放缓存碎片
        torch.cuda.empty_cache()
    writer.close()

if __name__ == "__main__":
    train()