import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional, Tuple
import time

from .model import MobileBertSpamClassifier
from .dataset import SpamDataset


class MobileBertTrainer:
    """
    Trainer class for MobileBERT spam classification model.
    """
    
    def __init__(self, model: MobileBertSpamClassifier, device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model (MobileBertSpamClassifier): The model to train
            device (str): Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training state
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
        # TensorBoard writer
        self.writer = None
        
    def setup_training(self, learning_rate: float = 2e-5, weight_decay: float = 0.01,
                      warmup_steps: int = 0, num_training_steps: int = 1000):
        """
        Setup training components (optimizer, scheduler, loss function).
        
        Args:
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            warmup_steps (int): Number of warmup steps for scheduler
            num_training_steps (int): Total number of training steps
        """
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        if warmup_steps > 0:
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def setup_tensorboard(self, log_dir: str = "runs"):
        """
        Setup TensorBoard logging.
        
        Args:
            log_dir (str): Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Add token_type_ids if available
            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Add token_type_ids if available
                token_type_ids = None
                if 'token_type_ids' in batch:
                    token_type_ids = batch['token_type_ids'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
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
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 5, save_dir: str = "models", 
              save_best: bool = True, early_stopping_patience: int = 3) -> Dict:
        """
        Main training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            save_dir (str): Directory to save models
            save_best (bool): Whether to save the best model
            early_stopping_patience (int): Number of epochs to wait before early stopping
            
        Returns:
            dict: Training history and best metrics
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup training if not already done
        if not hasattr(self, 'optimizer'):
            self.setup_training()
        
        # Early stopping
        patience_counter = 0
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}")
            print(f"Recall: {val_metrics['recall']:.4f}")
            print(f"F1-Score: {val_metrics['f1_score']:.4f}")
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
                self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
                self.writer.add_scalar('Metrics/F1', val_metrics['f1_score'], epoch)
            
            # Save best model
            if save_best and val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                patience_counter = 0
                
                # Save model
                best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
                self.save_model(best_model_path, epoch, val_metrics)
                self.best_model_path = best_model_path
                
                print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            self.save_checkpoint(checkpoint_path, epoch, train_loss, val_loss, train_acc, val_acc)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'best_model_path': self.best_model_path,
            'total_time': total_time
        }
        
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_model(self, file_path: str, epoch: int, metrics: Dict):
        """
        Save the model state.
        
        Args:
            file_path (str): Path to save the model
            epoch (int): Current epoch
            metrics (dict): Validation metrics
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'best_val_accuracy': self.best_val_accuracy
        }, file_path)
        
    def save_checkpoint(self, file_path: str, epoch: int, train_loss: float, 
                       val_loss: float, train_acc: float, val_acc: float):
        """
        Save a training checkpoint.
        
        Args:
            file_path (str): Path to save the checkpoint
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            train_acc (float): Training accuracy
            val_acc (float): Validation accuracy
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_val_accuracy': self.best_val_accuracy
        }, file_path)
    
    def load_checkpoint(self, file_path: str):
        """
        Load a training checkpoint.
        
        Args:
            file_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()
