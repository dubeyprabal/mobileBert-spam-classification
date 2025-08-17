# 🚀 MobileBERT Spam Classification - Demo Commands

## 📋 **Complete Demo Process - Copy & Paste Ready!**

### **1. Navigate to Project Directory & Activate Environment**
```bash
cd ~/Personal/spam-classification
source .venv/bin/activate
cd mobilebert_spam_classifier
```

### **2. Test Setup (Verify Everything Works)**
```bash
python test_setup.py
```
**Expected Output**: All 4 tests should pass ✅

### **3. Train the Model with Real SMS Dataset**
```bash
python train.py --data_path data/sms_spam_dataset.csv --epochs 5 --batch_size 8 --max_length 128 --learning_rate 1e-4
```
**Expected Output**: Training progress with real-time metrics

### **4. Check What Models Were Saved**
```bash
ls -la models/
```
**Expected Output**: List of saved model checkpoints

### **5. Evaluate the Best Model**
```bash
python evaluate.py --model_path models/best_model_epoch_X.pth --test_data data/sms_spam_dataset.csv --plot_results --save_predictions
```
**Note**: Replace `X` with the actual epoch number from step 4

### **6. Interactive Testing (Optional)**
```bash
python evaluate.py --model_path models/best_model_epoch_X.pth --test_data data/sms_spam_dataset.csv
```
**This will let you type messages and see real-time predictions**

---

## 🔧 **Alternative Training Commands (If Memory Issues)**

### **Memory-Optimized Training:**
```bash
python train.py --data_path data/sms_spam_dataset.csv --epochs 3 --batch_size 4 --max_length 64 --learning_rate 1e-4
```

### **Quick Demo Training (2 epochs):**
```bash
python train.py --data_path data/sms_spam_dataset.csv --epochs 2 --batch_size 8 --max_length 128 --learning_rate 1e-4
```

---

## 🎯 **One-Line Demo (Copy & Paste Everything)**
```bash
cd ~/Personal/spam-classification && source .venv/bin/activate && cd mobilebert_spam_classifier && python train.py --data_path data/sms_spam_dataset.csv --epochs 3 --batch_size 8 --max_length 128 --learning_rate 1e-4
```

---

## 📊 **Expected Results & Timeline**

### **Training Phase:**
- **Dataset Loading**: 5,572 SMS messages loaded
- **Model Creation**: MobileBERT with 24.7M parameters
- **Training Time**: 10-30 minutes (depending on settings)
- **Final Accuracy**: 90%+ (much better than synthetic data!)

### **Output Files:**
- **Models**: `models/best_model_epoch_X.pth`
- **Training History**: `models/training_history.png`
- **Evaluation Results**: `evaluation_results/` folder
- **Plots**: Confusion matrix, ROC curves, training curves

---

## 🔍 **Troubleshooting Commands**

### **If Training is Slow:**
```bash
python train.py --data_path data/sms_spam_dataset.csv --epochs 2 --batch_size 4 --max_length 64 --learning_rate 1e-4
```

### **If Memory Issues:**
```bash
python train.py --data_path data/sms_spam_dataset.csv --epochs 2 --batch_size 2 --max_length 32 --learning_rate 1e-4
```

### **Check System Resources:**
```bash
nvidia-smi  # If using GPU
htop        # CPU and memory usage
```

---

## 📱 **Dataset Information**

### **SMS Spam Collection Dataset:**
- **Total Messages**: 5,572
- **Ham (Legitimate)**: 4,825 (86.6%)
- **Spam**: 747 (13.4%)
- **Format**: CSV with `label` (0=ham, 1=spam) and `text` columns
- **Quality**: Real-world SMS data, professionally curated

---

## 🎉 **Demo Success Indicators**

✅ **Setup Test**: All 4 tests pass  
✅ **Training**: Loss decreases, accuracy increases  
✅ **Model Saved**: Checkpoint files in `models/` folder  
✅ **Evaluation**: 90%+ accuracy on test set  
✅ **Visualizations**: Training curves and performance plots generated  

---

## 💡 **Pro Tips for Demo**

1. **Start with 2-3 epochs** for quick demo
2. **Use batch_size 8** for good balance of speed/memory
3. **Monitor training progress** - loss should decrease
4. **Save the best model** for evaluation
5. **Generate plots** for impressive visual results

---

**Happy Demo! 🚀**

*Last Updated: August 11, 2025*
