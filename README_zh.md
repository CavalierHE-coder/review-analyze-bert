# 中文购物评论情感分析(BERT)

[English](README.md) | 简体中文

基于 `bert-base-chinese` 微调的中文购物评论情感二分类模型,输入一条评论,输出**正面 / 负面**及置信度。

## 项目特点

- 使用 HuggingFace Transformers 加载 `bert-base-chinese`,取 `[CLS]` 向量接分类头做二分类
- 使用 `BCEWithLogitsLoss`,模型直接输出 logits,预测时再做 sigmoid
- 训练支持 bf16 混合精度、梯度累积,在 8GB 显存的笔记本 GPU 上即可训练
- 使用 TensorBoard 记录训练 loss
- 评估脚本输出 Accuracy、Precision、Recall、F1 和混淆矩阵
- 提供命令行交互式预测

## 项目结构

```
review_analyze_bert/
├── data/
│   ├── raw/                  # 原始数据(online_shopping_10_cats.csv)
│   └── processed/            # 预处理后的 train / test(save_to_disk 格式)
├── pretrained/
│   └── bert-base-chinese/    # 预训练模型(需自行下载)
├── models/
│   └── model.pt              # 训练得到的权重(需自行训练或下载)
├── logs/                     # TensorBoard 日志
├── src/
│   ├── config.py             # 路径与超参数配置
│   ├── process.py            # 数据预处理(清洗、划分、分词、保存)
│   ├── dataset.py            # Dataset / DataLoader
│   ├── model.py              # ReviewAnalyzeBert 模型定义
│   ├── train.py              # 模型训练
│   ├── evaluate.py           # 模型评估(Accuracy / Precision / Recall / F1)
│   └── predict.py            # 交互式预测
├── requirements.txt
└── README.md
```

> 目录名以你本地 `config.py` 中的实际配置为准。

## 环境依赖

- Python 3.10+
- PyTorch(NVIDIA RTX 50 系显卡需要 CUDA 12.8 及以上的构建)
- transformers
- datasets
- tensorboard
- tqdm

安装:

```bash
pip install -r requirements.txt
```

> PyTorch 请根据显卡和 CUDA 版本,到 pytorch.org 选择对应的安装命令单独安装。

## 数据集

使用公开的中文电商评论数据集 `online_shopping_10_cats.csv`(10 个品类的商品评论,标签 0 为负面、1 为正面)。

请将文件放到 `data/raw/` 目录下。数据版权归原作者所有,本仓库不包含数据文件。

## 预训练模型

从 HuggingFace 下载 `bert-base-chinese`,放到 `pretrained/bert-base-chinese/`:

```bash
huggingface-cli download bert-base-chinese --local-dir pretrained/bert-base-chinese
```

## 使用方法

按顺序运行:

### 1. 数据预处理

```bash
python src/process.py
```

完成过滤空评论、分层划分训练/测试集(8:2)、分词(截断/填充到 `SEQ_LEN`),并保存到 `data/processed/`。

> 修改 `SEQ_LEN` 后需要重新运行此步骤,训练读取的是已保存的分词结果。

### 2. 模型训练

```bash
python src/train.py
```

训练过程中会把训练 loss 最低的权重保存为 `models/model.pt`。

查看训练曲线:

```bash
tensorboard --logdir ./logs
```

浏览器访问 http://localhost:6006 。

### 3. 模型评估

```bash
python src/evaluate.py
```

用 `models/model.pt` 在测试集上评估,输出样本总数、混淆矩阵(TP / FP / FN / TN)、Accuracy、Precision、Recall 和 F1(以正面为正类)。

### 4. 预测

```bash
python src/predict.py
```

示例:

```
请输入评价(输入q或者quit退出):
> 质量很好,物流也很快
正面评价(置信度:0.98)
> 太差了,用两天就坏了
负面评价(置信度:0.97)
```

## 主要配置

| 参数 | 说明 | 当前取值 |
|---|---|---|
| `SEQ_LEN` | 序列最大长度 | 64 |
| `BATCH_SIZE` | 批大小 | 32 |
| `ACCUM_STEPS` | 梯度累积步数 | 见 `train.py` |
| `LEARNING_RATE` | 学习率 | 见 `config.py` |
| `EPOCHS` | 训练轮数 | 见 `config.py` |

## 实验结果

| 指标 | 测试集    |
|---|--------|
| Accuracy | 0.9446 |
| Precision | 0.9508 |
| Recall | 0.9389 |
| F1 | 0.9448 |

训练环境:笔记本 RTX 5060(8GB),bf16 混合精度,约 15 it/s(batch 32,序列长度 64)。

## 模型权重

训练好的权重可从以下地址下载,放到 `models/model.pt`:

- HuggingFace:待补充链接

## 遇到过的问题

- **训练特别慢**:`tokenizer` 的参数名应为 `max_length`,写成 `max_len` 会被静默忽略,序列长度退回 512,导致训练极慢且显存溢出。
- **修改序列长度后无效**:`train.py` 读取的是磁盘上已保存的数据,需要重新运行 `process.py`。

## License

MIT
