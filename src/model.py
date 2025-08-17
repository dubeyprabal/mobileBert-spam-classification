import torch
import torch.nn as nn
from transformers import MobileBertModel, MobileBertTokenizer


class MobileBertSpamClassifier(nn.Module):
    """
    MobileBERT-based spam classifier with a classification head.
    """
    
    def __init__(self, model_name="google/mobilebert-uncased", num_classes=2, dropout=0.1):
        """
        Initialize the MobileBERT spam classifier.
        
        Args:
            model_name (str): Pre-trained MobileBERT model name
            num_classes (int): Number of output classes (2 for spam/ham)
            dropout (float): Dropout rate for the classification head
        """
        super(MobileBertSpamClassifier, self).__init__()
        
        # Load pre-trained MobileBERT model
        self.mobilebert = MobileBertModel.from_pretrained(model_name)
        
        # Get the hidden size from the model
        hidden_size = self.mobilebert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Freeze MobileBERT layers (optional - can be unfrozen for fine-tuning)
        # self._freeze_mobilebert()
    
    def _freeze_mobilebert(self):
        """Freeze MobileBERT parameters to prevent updates during training."""
        for param in self.mobilebert.parameters():
            param.requires_grad = False
    
    def unfreeze_mobilebert(self):
        """Unfreeze MobileBERT parameters for fine-tuning."""
        for param in self.mobilebert.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            torch.Tensor: Classification logits
        """
        # Get MobileBERT outputs
        outputs = self.mobilebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(cls_output)
        
        return logits
    
    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Get embeddings from MobileBERT without classification head.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            torch.Tensor: [CLS] token embeddings
        """
        with torch.no_grad():
            outputs = self.mobilebert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return outputs.last_hidden_state[:, 0, :]


def get_tokenizer(model_name="google/mobilebert-uncased"):
    """
    Get the tokenizer for the specified MobileBERT model.
    
    Args:
        model_name (str): Pre-trained MobileBERT model name
        
    Returns:
        MobileBertTokenizer: Tokenizer instance
    """
    return MobileBertTokenizer.from_pretrained(model_name)


def create_model(model_name="google/mobilebert-uncased", num_classes=2, dropout=0.1):
    """
    Factory function to create a MobileBERT spam classifier.
    
    Args:
        model_name (str): Pre-trained MobileBERT model name
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
        
    Returns:
        MobileBertSpamClassifier: Model instance
    """
    return MobileBertSpamClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout
    )
