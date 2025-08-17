import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import MobileBertTokenizer


class SpamDataset(Dataset):
    """
    Custom dataset for spam/ham classification using MobileBERT tokenization.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: MobileBertTokenizer, 
                 max_length: int = 128):
        """
        Initialize the dataset.
        
        Args:
            texts (List[str]): List of text messages
            labels (List[int]): List of labels (0 for ham, 1 for spam)
            tokenizer (MobileBertTokenizer): MobileBERT tokenizer
            max_length (int): Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate inputs
        assert len(texts) == len(labels), "Texts and labels must have the same length"
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove the batch dimension
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].flatten()
        
        return item


class SpamDataProcessor:
    """
    Utility class for processing spam/ham data from various sources.
    """
    
    @staticmethod
    def load_from_csv(file_path: str, text_column: str = 'text', label_column: str = 'label'):
        """
        Load data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            tuple: (texts, labels)
        """
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return texts, labels
    
    @staticmethod
    def load_from_dataframe(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label'):
        """
        Load data from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            text_column (str): Name of the text column
            label_column (str): Name of the label column
            
        Returns:
            tuple: (texts, labels)
        """
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in DataFrame")
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        return texts, labels
    
    @staticmethod
    def create_sample_data():
        """
        Create sample spam/ham data for testing purposes.
        
        Returns:
            tuple: (texts, labels)
        """
        sample_texts = [
            "Hi, how are you doing?",
            "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
            "Hello, this is a legitimate message from your bank.",
            "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
            "Meeting tomorrow at 3 PM in conference room A.",
            "CONGRATULATIONS! You've been selected to receive a $1000 gift card!",
            "Please review the attached documents before the meeting.",
            "Limited time offer! Buy now and get 50% off!",
            "Your order #12345 has been shipped and will arrive tomorrow.",
            "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
        ]
        
        sample_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0: ham, 1: spam
        
        return sample_texts, sample_labels


def create_dataset(texts: List[str], labels: List[int], tokenizer: MobileBertTokenizer, 
                  max_length: int = 128) -> SpamDataset:
    """
    Factory function to create a SpamDataset.
    
    Args:
        texts (List[str]): List of text messages
        labels (List[int]): List of labels
        tokenizer (MobileBertTokenizer): MobileBERT tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        SpamDataset: Dataset instance
    """
    return SpamDataset(texts, labels, tokenizer, max_length)
