import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation import DataPreprocessor, SignLanguageDataset
from model import HandKeypointCNN

class SignLanguageTrainer:
    def __init__(self, data_dir="../data/processed_data", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir
        
        # Load processed data and metadata
        self.load_data()
        
        # Initialize model
        self.initialize_model()
        
        # Training components
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()
        self.uncertainty_criterion = nn.BCELoss()
        
        print(f"Trainer initialized on {self.device}")
        print(f"Parameters: {self.count_parameters():,}")
    
    def load_data(self):
        """Load processed data"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Processed data directory '{self.data_dir}' not found. Run data preprocessing first.")
        
        print(f"Loading data from {self.data_dir}...")
        
        # Load data arrays
        self.X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        self.X_val = np.load(os.path.join(self.data_dir, "X_val.npy"))
        self.X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        self.y_train = np.load(os.path.join(self.data_dir, "y_train.npy"))
        self.y_val = np.load(os.path.join(self.data_dir, "y_val.npy"))
        self.y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))
        
        # Load metadata
        with open(os.path.join(self.data_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # Create datasets and loaders
        self.train_dataset = SignLanguageDataset(self.X_train, self.y_train)
        self.val_dataset = SignLanguageDataset(self.X_val, self.y_val)
        self.test_dataset = SignLanguageDataset(self.X_test, self.y_test)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        print(f"✓ Loaded {len(self.train_dataset)} training samples")
        print(f"✓ Loaded {len(self.val_dataset)} validation samples")
        print(f"✓ Loaded {len(self.test_dataset)} test samples")
        print(f"✓ {self.metadata['num_classes']} classes total")
    
    def initialize_model(self):
        """Initialize the model based on type"""
        num_classes = self.metadata['num_classes']
        
        self.model = HandKeypointCNN(num_classes=num_classes)
        
        self.model.to(self.device)
        self.unknown_class_idx = self.metadata['unknown_class_index']
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            class_logits, uncertainty = self.model(data)
            
            # Classification loss
            class_loss = self.criterion(class_logits, target)
            
            # Uncertainty loss (1 for unknown class, 0 for known classes)
            uncertainty_targets = (target == self.unknown_class_idx).float().unsqueeze(1)
            uncertainty_loss = self.uncertainty_criterion(uncertainty, uncertainty_targets)
            
            # Combined loss
            total_loss_batch = class_loss + 0.1 * uncertainty_loss
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            _, predicted = class_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        self.scheduler.step()
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def evaluate_validation(self, uncertainty_threshold=0.5):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.squeeze().to(self.device)
                
                class_logits, uncertainty = self.model(data)
                
                # Classification loss
                class_loss = self.criterion(class_logits, target)
                uncertainty_targets = (target == self.unknown_class_idx).float().unsqueeze(1)
                uncertainty_loss = self.uncertainty_criterion(uncertainty, uncertainty_targets)
                
                total_loss += (class_loss + 0.1 * uncertainty_loss).item()
                
                # Make predictions
                _, predicted = class_logits.max(1)
                high_uncertainty = uncertainty.squeeze() > uncertainty_threshold
                predicted[high_uncertainty] = self.unknown_class_idx
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def evaluate(self, uncertainty_threshold=0.5):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.squeeze().to(self.device)
                
                class_logits, uncertainty = self.model(data)
                
                # Classification loss
                class_loss = self.criterion(class_logits, target)
                uncertainty_targets = (target == self.unknown_class_idx).float().unsqueeze(1)
                uncertainty_loss = self.uncertainty_criterion(uncertainty, uncertainty_targets)
                
                total_loss += (class_loss + 0.1 * uncertainty_loss).item()
                
                # Make predictions considering uncertainty
                _, predicted = class_logits.max(1)
                
                # If uncertainty is high, predict unknown class
                high_uncertainty = uncertainty.squeeze() > uncertainty_threshold
                predicted[high_uncertainty] = self.unknown_class_idx
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_uncertainties.extend(uncertainty.squeeze().cpu().numpy())
        
        return (total_loss / len(self.test_loader), 
                100. * correct / total, 
                np.array(all_predictions), 
                np.array(all_targets),
                np.array(all_uncertainties))
    
    def train(self, epochs=100, save_dir="../models"):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n=== Training CNN Model ===")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Save directory: {save_dir}")
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        best_model_path = os.path.join(save_dir, f'best_cnn_model.pth')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.evaluate_validation()
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Save best model based on VALIDATION accuracy (not test)
            if val_acc > best_val_acc:  # Change this variable name
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'metadata': self.metadata,
                    'epoch': epoch,
                    'val_accuracy': val_acc
                }, best_model_path)
            
            if epoch % 1 == 0:
                print(f'Epoch {epoch}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
        print(f"✓ Best model saved: {best_model_path}")
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies, 
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
        }
        
        history_path = os.path.join(save_dir, f'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Training history saved: {history_path}")
        
        return history
    
    def final_evaluation(self, save_dir="../models"):
        """Comprehensive final evaluation"""
        print(f"\n=== Final Evaluation ===")
        
        test_loss, test_acc, predictions, targets, uncertainties = self.evaluate()
        
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = self.metadata['class_names'] + ['unknown']
        report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
        
        print("\nPer-class Performance:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Save evaluation results
        eval_results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'classification_report': report,
            'num_parameters': self.count_parameters()
        }
        
        eval_path = os.path.join(save_dir, f'evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"✓ Evaluation results saved: {eval_path}")
        
        return eval_results
    
    def plot_training_history(self, history, save_dir="../models"):
        """Plot and save training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Train Loss', color='blue')
        ax1.plot(history['val_losses'], label='Val Loss', color='red')
        ax1.set_title(f'CNN Training and Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['train_accuracies'], label='Train Accuracy', color='blue')
        ax2.plot(history['val_accuracies'], label='Val Accuracy', color='red')
        ax2.set_title(f'CNN Training and Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Training plots saved: {plot_path}")

def main():
    """Main training pipeline"""
    print("=== Sign Language Model Training Pipeline ===")
    
    # Check if processed data exists
    if not os.path.exists("../data/processed_data"):
        print("❌ Processed data not found!")
        print("Please run data preprocessing first:")
        print("python data_preparation.py")
        return
    
    # Train different models
    results = {}
    
    print(f"\n{'='*50}")
    print(f"Training CNN Model")
    print(f"{'='*50}")
    
    try:
        # Initialize trainer
        trainer = SignLanguageTrainer()
        
        # Train model
        history = trainer.train(epochs=100)
        
        # Final evaluation
        eval_results = trainer.final_evaluation()
        
        # Plot training history
        trainer.plot_training_history(history)
        
        results = {
            'test_accuracy': eval_results['test_accuracy'],
            'num_parameters': eval_results['num_parameters']
        }
        
    except Exception as e:
        print(f"❌ Error training CNN: {e}")
    
    # Compare results
    if results:
        print(f"\n{'='*50}")
        print("Model Comparison Summary")
        print(f"{'='*50}")
        
        print(f"CNN: {results['test_accuracy']:.2f}% accuracy, {results['num_parameters']:,} parameters")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 Best Model: {best_model[0].upper()} with {best_model[1]['test_accuracy']:.2f}% accuracy")

if __name__ == "__main__":
    main()