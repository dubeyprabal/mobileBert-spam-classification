#!/usr/bin/env python3
"""
Main training script for MobileBERT spam classification model.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_model, get_tokenizer
from src.dataset import SpamDataset, SpamDataProcessor, create_dataset
from src.trainer import MobileBertTrainer
from src.utils import (
    split_data, create_sample_dataset, plot_training_history,
    create_experiment_log, print_model_summary
)


def main():
    parser = argparse.ArgumentParser(description='Train MobileBERT spam classifier')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to CSV file with text and label columns')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the text column in CSV')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of the label column in CSV')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use sample data instead of loading from file')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples when using sample data')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='google/mobilebert-uncased',
                       help='Pre-trained MobileBERT model name')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for classification head')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=32,
                       help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Number of warmup steps')
    parser.add_argument('--early_stopping_patience', type=int, default=3,
                       help='Early stopping patience')
    
    # Data split arguments
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of remaining data for validation set')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--experiment_name', type=str, default='mobilebert_spam_classifier',
                       help='Name for the experiment')
    parser.add_argument('--enable_tensorboard', action='store_true',
                       help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_dir', type=str, default='runs',
                       help='Directory for TensorBoard logs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, or None for auto)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("MobileBERT Spam Classification Training")
    print("=" * 60)
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load or create data
    if args.use_sample_data:
        print("Creating sample dataset...")
        df = create_sample_dataset(num_samples=args.num_samples, random_state=args.random_state)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        print(f"Created {len(texts)} sample messages")
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        texts, labels = SpamDataProcessor.load_from_csv(
            args.data_path, args.text_column, args.label_column
        )
        print(f"Loaded {len(texts)} messages from {args.data_path}")
    else:
        print("No data source specified. Creating sample dataset...")
        df = create_sample_dataset(num_samples=args.num_samples, random_state=args.random_state)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        print(f"Created {len(texts)} sample messages")
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
        texts, labels, args.test_size, args.val_size, args.random_state
    )
    
    print(f"Train set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = create_dataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = create_dataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.val_batch_size, 
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.val_batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("Creating MobileBERT model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=2,
        dropout=args.dropout
    )
    
    # Print model summary
    print_model_summary(model)
    
    # Create trainer
    print("Setting up trainer...")
    trainer = MobileBertTrainer(model, device=device)
    
    # Setup training
    num_training_steps = len(train_loader) * args.epochs
    trainer.setup_training(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Setup TensorBoard if enabled
    if args.enable_tensorboard:
        print(f"Setting up TensorBoard logging in {args.tensorboard_dir}")
        trainer.setup_tensorboard(args.tensorboard_dir)
    
    # Training configuration
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'val_batch_size': args.val_batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'early_stopping_patience': args.early_stopping_patience,
        'test_size': args.test_size,
        'val_size': args.val_size,
        'random_state': args.random_state,
        'device': device,
        'num_train_samples': len(train_texts),
        'num_val_samples': len(val_texts),
        'num_test_samples': len(test_texts)
    }
    
    # Start training
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=args.save_dir,
            save_best=True,
            early_stopping_patience=args.early_stopping_patience
        )
        
        print("\nTraining completed successfully!")
        
        # Plot training history
        print("Plotting training history...")
        plot_training_history(history, save_path=os.path.join(args.save_dir, 'training_history.png'))
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc, test_metrics = trainer.validate(test_loader)
        
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        
        # Create experiment log
        results = {
            'training_history': history,
            'test_metrics': test_metrics,
            'best_model_path': trainer.best_model_path
        }
        
        experiment_dir = create_experiment_log(
            args.experiment_name, config, results, log_dir="experiments"
        )
        
        print(f"\nExperiment logged to: {experiment_dir}")
        print(f"Best model saved to: {trainer.best_model_path}")
        
        if args.enable_tensorboard:
            print(f"TensorBoard logs available in: {args.tensorboard_dir}")
            print("Run 'tensorboard --logdir runs' to view training progress")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.close()
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.close()
        raise
    finally:
        trainer.close()
    
    print("\n" + "=" * 60)
    print("Training script completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
