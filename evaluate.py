#!/usr/bin/env python3
"""
Evaluation script for trained MobileBERT spam classification model.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_model, get_tokenizer
from src.dataset import SpamDataset, SpamDataProcessor, create_dataset
from src.utils import (
    plot_confusion_matrix, plot_roc_curve, save_predictions,
    create_sample_dataset
)


def load_model(model_path: str, model_name: str = 'google/mobilebert-uncased', 
               device: str = None) -> tuple:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        model_name (str): Pre-trained MobileBERT model name
        device (str): Device to use
        
    Returns:
        tuple: (model, tokenizer, checkpoint_info)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Create model
    model = create_model(model_name=model_name, num_classes=2)
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get tokenizer
    tokenizer = get_tokenizer(model_name)
    
    return model, tokenizer, checkpoint


def predict_single_text(model: torch.nn.Module, tokenizer, text: str, 
                       max_length: int = 128, device: str = None) -> tuple:
    """
    Predict spam/ham for a single text.
    
    Args:
        model (torch.nn.Module): Trained model
        tokenizer: Tokenizer instance
        text (str): Input text
        max_length (int): Maximum sequence length
        device (str): Device to use
        
    Returns:
        tuple: (predicted_label, confidence_score, raw_logits)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    
    # Tokenize text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Add token_type_ids if available
    token_type_ids = None
    if 'token_type_ids' in encoding:
        token_type_ids = encoding['token_type_ids'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        logits = outputs
        probabilities = F.softmax(logits, dim=1)
        
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence_score = probabilities[0][predicted_label].item()
        raw_logits = logits[0].cpu().numpy()
    
    return predicted_label, confidence_score, raw_logits


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, 
                  device: str = None) -> tuple:
    """
    Evaluate model on test data.
    
    Args:
        model (torch.nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (str): Device to use
        
    Returns:
        tuple: (loss, accuracy, metrics_dict, predictions_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_scores = []
    total_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Add token_type_ids if available
            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            scores = probabilities[:, 1]  # Probability of spam class
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': classification_report(all_labels, all_predictions)
    }
    
    predictions_dict = {
        'true_labels': all_labels,
        'predicted_labels': all_predictions,
        'scores': all_scores
    }
    
    return avg_loss, accuracy, metrics, predictions_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate MobileBERT spam classifier')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--model_name', type=str, default='google/mobilebert-uncased',
                       help='Pre-trained MobileBERT model name')
    
    # Data arguments
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data CSV file')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the text column in CSV')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of the label column in CSV')
    parser.add_argument('--use_sample_data', action='store_true',
                       help='Use sample data for evaluation')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of samples when using sample data')
    
    # Evaluation arguments
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Evaluation batch size')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions to CSV')
    parser.add_argument('--plot_results', action='store_true',
                       help='Generate evaluation plots')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, or None for auto)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MobileBERT Spam Classification Evaluation")
    print("=" * 60)
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer, checkpoint = load_model(
        args.model_path, args.model_name, device
    )
    
    print(f"Model loaded successfully!")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_accuracy', 'Unknown')}")
    
    # Load or create test data
    if args.use_sample_data:
        print("Creating sample test dataset...")
        df = create_sample_dataset(num_samples=args.num_samples, random_state=42)
        test_texts = df['text'].tolist()
        test_labels = df['label'].tolist()
        print(f"Created {len(test_texts)} sample test messages")
    elif args.test_data:
        print(f"Loading test data from {args.test_data}...")
        test_texts, test_labels = SpamDataProcessor.load_from_csv(
            args.test_data, args.text_column, args.label_column
        )
        print(f"Loaded {len(test_texts)} test messages")
    else:
        print("No test data specified. Creating sample test dataset...")
        df = create_sample_dataset(num_samples=args.num_samples, random_state=42)
        test_texts = df['text'].tolist()
        test_labels = df['label'].tolist()
        print(f"Created {len(test_texts)} sample test messages")
    
    # Create test dataset and loader
    print("Creating test dataset...")
    test_dataset = create_dataset(test_texts, test_labels, tokenizer, args.max_length)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc, test_metrics, predictions = evaluate_model(
        model, test_loader, device
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(test_metrics['classification_report'])
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_file = os.path.join(args.output_dir, 'predictions.csv')
        save_predictions(
            test_texts,
            predictions['true_labels'],
            predictions['predicted_labels'],
            predictions['scores'],
            predictions_file
        )
    
    # Generate plots if requested
    if args.plot_results:
        print("Generating evaluation plots...")
        
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            predictions['true_labels'], 
            predictions['predicted_labels'],
            save_path=cm_path
        )
        
        # ROC curve
        roc_path = os.path.join(args.output_dir, 'roc_curve.png')
        plot_roc_curve(
            predictions['true_labels'],
            predictions['scores'],
            save_path=roc_path
        )
    
    # Interactive testing
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)
    print("Enter text messages to classify (type 'quit' to exit):")
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Predict
            predicted_label, confidence, _ = predict_single_text(
                model, tokenizer, text, args.max_length, device
            )
            
            label_name = "SPAM" if predicted_label == 1 else "HAM"
            print(f"Prediction: {label_name}")
            print(f"Confidence: {confidence:.4f}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Save evaluation summary
    summary = {
        'model_path': args.model_path,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1_score': test_metrics['f1_score'],
        'num_test_samples': len(test_texts),
        'device': device
    }
    
    summary_file = os.path.join(args.output_dir, 'evaluation_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation summary saved to: {summary_file}")
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
