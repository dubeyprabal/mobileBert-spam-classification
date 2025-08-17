import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import torch
from typing import List, Tuple, Dict, Any
import json
import os
from datetime import datetime


def split_data(texts: List[str], labels: List[int], test_size: float = 0.2, 
               val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        texts (List[str]): List of text messages
        labels (List[int]): List of labels
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of remaining data for validation set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
    """
    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: separate validation set from training set
    val_size_adjusted = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_val_labels
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def create_sample_dataset(num_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a sample spam/ham dataset for testing purposes.
    
    Args:
        num_samples (int): Number of samples to create
        random_state (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    np.random.seed(random_state)
    
    # Sample ham messages (legitimate)
    ham_messages = [
        "Hi, how are you doing?",
        "Meeting tomorrow at 3 PM in conference room A.",
        "Please review the attached documents before the meeting.",
        "Your order #12345 has been shipped and will arrive tomorrow.",
        "Hello, this is a legitimate message from your bank.",
        "The project deadline has been extended to next Friday.",
        "Can you please send me the updated report?",
        "Thanks for your help with the presentation.",
        "I'll be out of office next week.",
        "The team meeting is scheduled for Monday morning."
    ]
    
    # Sample spam messages
    spam_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
        "CONGRATULATIONS! You've been selected to receive a $1000 gift card!",
        "Limited time offer! Buy now and get 50% off!",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "URGENT! Your account has been compromised. Click here to verify.",
        "FREE RINGTONE text POLY to 87121 for a poly or text RING to 87121 for a true tone!",
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
        "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!"
    ]
    
    # Generate dataset
    texts = []
    labels = []
    
    # Add ham messages
    num_ham = int(num_samples * 0.7)  # 70% ham, 30% spam
    for _ in range(num_ham):
        text = np.random.choice(ham_messages)
        # Add some variation
        if np.random.random() > 0.5:
            text += f" {np.random.randint(1000, 9999)}"
        texts.append(text)
        labels.append(0)
    
    # Add spam messages
    num_spam = num_samples - num_ham
    for _ in range(num_spam):
        text = np.random.choice(spam_messages)
        # Add some variation
        if np.random.random() > 0.5:
            text += f" {np.random.randint(1000, 9999)}"
        texts.append(text)
        labels.append(1)
    
    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    return df


def plot_training_history(history: Dict[str, List], save_path: str = None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss', marker='o')
    ax1.plot(history['val_losses'], label='Validation Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         save_path: str = None, normalize: bool = True):
    """
    Plot confusion matrix.
    
    Args:
        y_true (List[int]): True labels
        y_pred (List[int]): Predicted labels
        save_path (str): Path to save the plot (optional)
        normalize (bool): Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                cmap='Blues', cbar=True, square=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Ham', 'Spam'])
    plt.yticks([0.5, 1.5], ['Ham', 'Spam'])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true: List[int], y_scores: List[float], 
                   save_path: str = None):
    """
    Plot ROC curve.
    
    Args:
        y_true (List[int]): True labels
        y_scores (List[float]): Prediction scores (probabilities)
        save_path (str): Path to save the plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_predictions(texts: List[str], true_labels: List[int], 
                    predicted_labels: List[int], predicted_scores: List[float],
                    save_path: str):
    """
    Save predictions to a CSV file.
    
    Args:
        texts (List[str]): Input texts
        true_labels (List[int]): True labels
        predicted_labels (List[int]): Predicted labels
        predicted_scores (List[float]): Prediction scores
        save_path (str): Path to save the CSV file
    """
    df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predicted_labels,
        'predicted_score': predicted_scores,
        'correct': [t == p for t, p in zip(true_labels, predicted_labels)]
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def create_experiment_log(experiment_name: str, config: Dict[str, Any], 
                         results: Dict[str, Any], log_dir: str = "experiments"):
    """
    Create an experiment log with configuration and results.
    
    Args:
        experiment_name (str): Name of the experiment
        config (dict): Configuration parameters
        results (dict): Results and metrics
        log_dir (str): Directory to save experiment logs
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save results
    results_path = os.path.join(experiment_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config_file": config_path,
        "results_file": results_path
    }
    
    summary_path = os.path.join(experiment_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment log created at: {experiment_dir}")
    return experiment_dir


def load_experiment_log(experiment_dir: str) -> Tuple[Dict, Dict]:
    """
    Load experiment configuration and results.
    
    Args:
        experiment_dir (str): Path to experiment directory
        
    Returns:
        tuple: (config, results)
    """
    config_path = os.path.join(experiment_dir, "config.json")
    results_path = os.path.join(experiment_dir, "results.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return config, results


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
