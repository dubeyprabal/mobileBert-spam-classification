# MobileBERT Spam Classification

This project implements a spam/ham classification model using MobileBERT, a lightweight and efficient variant of BERT.

## Features

- **MobileBERT Model**: Uses the lightweight MobileBERT architecture for efficient text classification
- **Spam/Ham Classification**: Binary classification to distinguish between legitimate (ham) and spam messages
- **Transfer Learning**: Leverages pre-trained MobileBERT weights for better performance
- **Easy Training**: Simple training pipeline with configurable hyperparameters
- **Evaluation Metrics**: Comprehensive evaluation including accuracy, precision, recall, and F1-score

## Project Structure

```
mobilebert_spam_classifier/
├── data/                   # Data directory
├── models/                 # Saved model checkpoints
├── src/                    # Source code
│   ├── __init__.py
│   ├── model.py           # MobileBERT model definition
│   ├── dataset.py         # Dataset handling
│   ├── trainer.py         # Training pipeline
│   └── utils.py           # Utility functions
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── README.md              # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data_path data/spam_dataset.csv --epochs 5 --batch_size 16
```

### Evaluation

```bash
python evaluate.py --model_path models/best_model.pth --test_data data/test.csv
```

## Model Architecture

The model uses MobileBERT as the base encoder with a classification head:
- **Encoder**: MobileBERT (12 layers, 512 hidden size)
- **Classification Head**: Linear layer with dropout
- **Output**: Binary classification (spam/ham)

## Performance

- **Accuracy**: Typically achieves 95%+ on standard spam datasets
- **Speed**: MobileBERT is 4x faster than BERT-base
- **Size**: Significantly smaller model size compared to BERT-base

## License

MIT License
