# Chinese Shopping Review Sentiment Analysis (BERT)

English | [简体中文](README_zh.md)

A binary sentiment classifier for Chinese shopping reviews, fine-tuned from `bert-base-chinese`. Given a review, it outputs **positive / negative** along with a confidence score.

## Features

- Loads `bert-base-chinese` with HuggingFace Transformers and uses the `[CLS]` vector with a classification head for binary classification
- Trained with `BCEWithLogitsLoss`: the model outputs logits directly, and sigmoid is applied at prediction time
- Training supports bf16 mixed precision and gradient accumulation, so it runs on a laptop GPU with 8GB of VRAM
- Training loss is logged with TensorBoard
- Evaluation script reports Accuracy, Precision, Recall, F1 and the confusion matrix
- Interactive command-line prediction

## Project Structure

```
review_analyze_bert/
├── data/
│   ├── raw/                  # Raw data (online_shopping_10_cats.csv)
│   └── processed/            # Preprocessed train / test sets (save_to_disk format)
├── pretrained/
│   └── bert-base-chinese/    # Pretrained model (download it yourself)
├── models/
│   └── model.pt              # Trained weights (train it yourself or download)
├── logs/                     # TensorBoard logs
├── src/
│   ├── config.py             # Paths and hyperparameter settings
│   ├── process.py            # Data preprocessing (cleaning, split, tokenization, saving)
│   ├── dataset.py            # Dataset / DataLoader
│   ├── model.py              # ReviewAnalyzeBert model definition
│   ├── train.py              # Model training
│   ├── evaluate.py           # Model evaluation (Accuracy / Precision / Recall / F1)
│   └── predict.py            # Interactive prediction
├── requirements.txt
└── README.md
```

> Directory names follow the actual settings in your local `config.py`.

## Requirements

- Python 3.10+
- PyTorch (NVIDIA RTX 50-series GPUs need a build for CUDA 12.8 or later)
- transformers
- datasets
- tensorboard
- tqdm

Install the dependencies:

```bash
pip install -r requirements.txt
```

> Install PyTorch separately by choosing the command that matches your GPU and CUDA version at pytorch.org.

## Dataset

This project uses the public Chinese e-commerce review dataset `online_shopping_10_cats.csv` (product reviews from 10 categories; label 0 is negative, 1 is positive).

Place the file in `data/raw/`. The dataset's copyright belongs to its original authors, and this repository does not include the data file.

## Pretrained Model

Download `bert-base-chinese` from HuggingFace and place it in `pretrained/bert-base-chinese/`:

```bash
huggingface-cli download bert-base-chinese --local-dir pretrained/bert-base-chinese
```

## Usage

Run in order:

### 1. Data Preprocessing

```bash
python src/process.py
```

This filters out empty reviews, performs a stratified train/test split (8:2), tokenizes the text (truncating/padding to `SEQ_LEN`), and saves the result to `data/processed/`.

> After changing `SEQ_LEN`, this step must be rerun, because training reads the tokenized results that were saved to disk.

### 2. Model Training

```bash
python src/train.py
```

During training, the weights with the lowest training loss are saved to `models/model.pt`.

View the training curves:

```bash
tensorboard --logdir ./logs
```

Open http://localhost:6006 in your browser.

### 3. Model Evaluation

```bash
python src/evaluate.py
```

Runs `models/model.pt` on the test set and prints the sample count, confusion matrix (TP / FP / FN / TN), Accuracy, Precision, Recall and F1, with positive as the positive class.

### 4. Prediction

```bash
python src/predict.py
```

Example:

```
Enter a review (type q or quit to exit):
> 质量很好,物流也很快
Positive review (confidence: 0.98)
> 太差了,用两天就坏了
Negative review (confidence: 0.97)
```

> Note: `predict.py` currently prints its prompts in Chinese; the English output above is for illustration.

## Key Settings

| Parameter | Description | Current Value |
|---|---|---|
| `SEQ_LEN` | Maximum sequence length | 64 |
| `BATCH_SIZE` | Batch size | 32 |
| `ACCUM_STEPS` | Gradient accumulation steps | See `train.py` |
| `LEARNING_RATE` | Learning rate | See `config.py` |
| `EPOCHS` | Number of epochs | See `config.py` |

## Results

| Metric | Test Set |
|---|----------|
| Accuracy | 0.9446   |
| Precision | 0.9508   |
| Recall | 0.9389   |
| F1 | 0.9448   |

Training environment: laptop RTX 5060 (8GB), bf16 mixed precision, about 15 it/s (batch size 32, sequence length 64).

## Model Weights

The trained weights can be downloaded from the address below and placed at `models/model.pt`:

- HuggingFace: link to be added

## Problems Encountered

- **Training was extremely slow**: the tokenizer argument should be `max_length`. Writing `max_len` is silently ignored, so the sequence length falls back to 512, which makes training extremely slow and overflows GPU memory.
- **Changing the sequence length had no effect**: `train.py` reads the data that was already saved to disk, so `process.py` needs to be rerun.

## License

MIT
