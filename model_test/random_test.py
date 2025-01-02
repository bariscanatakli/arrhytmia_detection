import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Style
init()

class ModelTester:
    def __init__(self):
        classes = [
    "SR", #Sinus Rhythm (SR): A normal rhythm originating from the sinus node.
    "AFIB", #Atrial Fibrillation (AFIB): A supraventricular tachyarrhythmia characterized by uncoordinated atrial activation with consequent deterioration of atrial mechanical function.
    "STACH", #Supraventricular Tachycardia (STACH): A tachyarrhythmia originating above the ventricles.
    "SARRH", #Sinus Arrhythmia (SARRH): A normal variation in heart rate caused by changes in the rate and depth of breathing.
    "PVC", #Premature Ventricular Contraction (PVC): A ventricular ectopic beat occurring earlier than the next expected sinus beat.
    "PAC", #Premature Atrial Contraction (PAC): An atrial ectopic beat occurring earlier than the next expected sinus beat.
    "AFLT", #Atrial Flutter (AFLT): A supraventricular tachyarrhythmia characterized by organized atrial depolarizations at a rate of 240-320 bpm.
    "SBRAD", #Sinus Bradycardia (SBRAD): A normal rhythm originating from the sinus node with a rate less than 60 bpm.
    "SVTAC", #Supraventricular Tachycardia (SVTAC): A tachyarrhythmia originating above the ventricles.
    "NORM", #Normal (NORM): A normal ECG signal.
     ]
        self.classes = classes
        self.data_dir = './'
        self.X_file = os.path.join(self.data_dir, 'preprocessed_X.npy')
        self.y_file = os.path.join(self.data_dir, 'preprocessed_y.npy')
        self.model_file = os.path.join(self.data_dir, 'best_model.keras')

    def load_data(self):
        if not os.path.exists(self.X_file) or not os.path.exists(self.y_file):
            raise FileNotFoundError("Data files not found. Please run the train_model script first.")
        self.X = np.load(self.X_file)
        self.y = np.load(self.y_file)

    def load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError("Model file not found")
        self.model = load_model(self.model_file)

    def select_random_records(self, n_samples=100):
        total_samples = len(self.X)
        random_indices = np.random.choice(total_samples, n_samples, replace=False)
        return self.X[random_indices], self.y[random_indices]

    def predict_and_analyze(self, n_samples=100):
        # Select random records
        X_sample, y_sample = self.select_random_records(n_samples)
        
        # Get predictions
        predictions = self.model.predict(X_sample)
        pred_binary = (predictions > 0.5).astype(np.int32)
        
        # Store results
        results = []
        
        print(f"\n{Fore.CYAN}Analyzing {n_samples} Random Records:{Style.RESET_ALL}\n")
        
        for i in range(n_samples):
            true_classes = [self.classes[j] for j in range(len(self.classes)) if y_sample[i][j] == 1]
            pred_classes = [self.classes[j] for j in range(len(self.classes)) if pred_binary[i][j] == 1]
            
            max_prob = max(predictions[i])
            confidence = "High" if max_prob > 0.8 else "Moderate" if max_prob > 0.6 else "Low"
            
            results.append({
                'true': true_classes,
                'predicted': pred_classes,
                'probabilities': predictions[i],
                'confidence': confidence
            })
            
            # Print individual record results
            if i < 100:  # Show detailed results for first 10 records
                print(f"\n{Fore.GREEN}Record #{i+1}:{Style.RESET_ALL}")
                print(f"True Classes: {', '.join(true_classes)}")
                print(f"Predicted: {', '.join(pred_classes)}")
                print(f"Confidence: {confidence}")
                print(f"Probabilities:")
                for cls, prob in zip(self.classes, predictions[i]):
                    if prob > 0.3:  # Only show significant probabilities
                        print(f"  {cls}: {prob:.3f}")
        
        # Calculate and show summary metrics
        self.show_summary_metrics(y_sample, pred_binary)
        
        return results

    def show_summary_metrics(self, y_true, y_pred_binary):
        print(f"\n{Fore.CYAN}Summary Metrics:{Style.RESET_ALL}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred_binary,
            target_names=self.classes
        ))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred_binary, axis=1))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes,
                   yticklabels=self.classes)
        plt.title('Confusion Matrix for Random Sample')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'random_sample_confusion_matrix.png'))
        plt.close()

def main():
    tester = ModelTester()
    try:
        print("Loading data...")
        tester.load_data()
        print("Loading model...")
        tester.load_model()
        
        # Analyze 100 random records
        tester.predict_and_analyze(n_samples=100)
        print("\nAnalysis complete. Results saved in model_test directory.")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()