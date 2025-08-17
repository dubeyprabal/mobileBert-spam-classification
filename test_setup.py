#!/usr/bin/env python3
"""
Test script to verify MobileBERT spam classification setup.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from src.model import create_model, get_tokenizer
        print("✓ Model module imported")
    except ImportError as e:
        print(f"✗ Model module import failed: {e}")
        return False
    
    try:
        from src.dataset import SpamDataset, SpamDataProcessor
        print("✓ Dataset module imported")
    except ImportError as e:
        print(f"✗ Dataset module import failed: {e}")
        return False
    
    try:
        from src.trainer import MobileBertTrainer
        print("✓ Trainer module imported")
    except ImportError as e:
        print(f"✗ Trainer module import failed: {e}")
        return False
    
    try:
        from src.utils import create_sample_dataset
        print("✓ Utils module imported")
    except ImportError as e:
        print(f"✗ Utils module import failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and basic functionality."""
    print("\nTesting model creation...")
    
    try:
        from src.model import create_model, get_tokenizer
        
        # Test tokenizer
        tokenizer = get_tokenizer('google/mobilebert-uncased')
        print("✓ Tokenizer loaded successfully")
        
        # Test model creation
        model = create_model('google/mobilebert-uncased', num_classes=2)
        print("✓ Model created successfully")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_dataset_creation():
    """Test dataset creation and processing."""
    print("\nTesting dataset creation...")
    
    try:
        from src.utils import create_sample_dataset, split_data
        
        # Create sample dataset
        df = create_sample_dataset(num_samples=100, random_state=42)
        print(f"✓ Sample dataset created with {len(df)} samples")
        
        # Test data splitting
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(
            texts, labels, test_size=0.2, val_size=0.1, random_state=42
        )
        
        print(f"✓ Data split successfully: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False


def test_training_setup():
    """Test training setup without actually training."""
    print("\nTesting training setup...")
    
    try:
        from src.model import create_model, get_tokenizer
        from src.utils import create_sample_dataset
        from src.dataset import create_dataset
        from src.trainer import MobileBertTrainer
        from torch.utils.data import DataLoader
        
        # Create minimal dataset
        df = create_sample_dataset(num_samples=50, random_state=42)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Create model and tokenizer
        tokenizer = get_tokenizer('google/mobilebert-uncased')
        model = create_model('google/mobilebert-uncased', num_classes=2)
        
        # Create dataset
        dataset = create_dataset(texts, labels, tokenizer, max_length=64)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create trainer
        trainer = MobileBertTrainer(model)
        trainer.setup_training(learning_rate=2e-5, weight_decay=0.01)
        
        print("✓ Training setup completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MobileBERT Spam Classification - Setup Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_dataset_creation,
        test_training_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Run: python example.py")
        print("2. Run: python train.py --use_sample_data")
        print("3. Run: python evaluate.py --model_path models/best_model_epoch_X.pth --use_sample_data")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
