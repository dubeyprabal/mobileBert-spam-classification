#!/usr/bin/env python3
"""
Simple example script demonstrating MobileBERT spam classification.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_model, get_tokenizer
from src.utils import create_sample_dataset, split_data
from src.dataset import create_dataset
from torch.utils.data import DataLoader
import torch


def main():
    print("=" * 60)
    print("MobileBERT Spam Classification Example")
    print("=" * 60)
    
    # Create sample data
    print("Creating sample dataset...")
    df = create_sample_dataset(num_samples=500, random_state=42)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    print(f"Created {len(texts)} sample messages")
    print(f"Ham messages: {len(df[df['label'] == 0])}")
    print(f"Spam messages: {len(df[df['label'] == 1])}")
    
    # Split data
    print("\nSplitting data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, test_size=0.2, val_size=0.1, random_state=42
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Load tokenizer and create model
    print("\nLoading tokenizer and creating model...")
    tokenizer = get_tokenizer('google/mobilebert-uncased')
    model = create_model('google/mobilebert-uncased', num_classes=2, dropout=0.1)
    
    # Create datasets
    print("Creating datasets...")
    max_length = 128
    train_dataset = create_dataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model created and moved to {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Output sample: {outputs[0][:5]}")
    
    print("\nExample completed successfully!")
    print("Run 'python train.py --use_sample_data' to train the model")
    print("Run 'python evaluate.py --model_path models/best_model_epoch_X.pth --use_sample_data' to evaluate")


if __name__ == "__main__":
    main()
