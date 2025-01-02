import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Dense, Dropout, LSTM, Add, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Custom callback to safely calculate metrics like precision, recall, and AUC
class SafeMetricsCallback(Callback):
    def __init__(self, validation_data, class_names):
        """
        Initialize callback with validation data and class names.
        """
        super().__init__()
        self.x_val, self.y_val = validation_data  # Validation data (features and labels)
        self.class_names = class_names  # List of class names
        
    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate metrics after each epoch and log them.
        """
        y_pred = self.model.predict(self.x_val, verbose=0)  # Predict on validation data
        y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
        
        metrics = {}
        
        # Calculate macro metrics
        metrics['precision'] = precision_score(self.y_val, y_pred_binary, average='macro', zero_division=0)
        metrics['recall'] = recall_score(self.y_val, y_pred_binary, average='macro', zero_division=0)
        metrics['f1'] = f1_score(self.y_val, y_pred_binary, average='macro', zero_division=0)
        
        # Calculate per-class AUC, if possible
        aucs = []
        for i, class_name in enumerate(self.class_names):
            if len(np.unique(self.y_val[:, i])) > 1:  # Ensure the class has positive and negative samples
                try:
                    auc = roc_auc_score(self.y_val[:, i], y_pred[:, i])
                    aucs.append(auc)
                except:
                    continue
        
        metrics['auc'] = np.mean(aucs) if aucs else 0.0  # Average AUC across all classes
        
        # Log metrics
        print(f"\nEpoch {epoch+1} Metrics:")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
        print(f"AUC: {metrics['auc']:.3f}")
        
        logs.update({k: v for k, v in metrics.items()})

# Function to define the model architecture
def create_model(input_shape, num_classes):
    """
    Define a 1D convolutional neural network with LSTM and attention for ECG classification.
    """
    inputs = Input(shape=input_shape)  # Input layer
    
    # First convolutional block
    x = Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Second convolutional block
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Residual block with shortcut connection
    x_shortcut = Conv1D(256, kernel_size=1, padding='same')(x)  # Shortcut path
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])  # Combine with shortcut
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # LSTM layer for sequential processing
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # Attention mechanism
    attention = Attention()([x, x])  # Self-attention
    x = GlobalAveragePooling1D()(attention)  # Pooling to create a fixed-length vector
    
    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)  # Output layer with sigmoid activation
    
    return Model(inputs=inputs, outputs=outputs)

# Main function to execute the entire workflow
def main():
    # File paths and dataset information
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Current directory
    dataset_folder = "dataset/"  # Folder containing dataset files
    path = os.path.join(current_dir, dataset_folder)

    # Check if the dataset file exists
    try:
        df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    except FileNotFoundError:
        print("Dataset file not found. Please download the dataset, extract it, and name the folder as 'dataset'.")
        return
    X_file = 'preprocessed_X.npy'  # File for preprocessed features
    y_file = 'preprocessed_y.npy'  # File for preprocessed labels
    classes = [
    "SR", #Sinus Rhythm (SR): A normal rhythm originating from the sinus node.
    "AFIB", #Atrial Fibrillation (AFIB): A supraventricular tachyarrhythmia characterized
    #by uncoordinated atrial activation with consequent deterioration of atrial mechanical function.
    "STACH", #Supraventricular Tachycardia (STACH): A tachyarrhythmia originating above the ventricles.
    "SARRH", #Sinus Arrhythmia (SARRH): A normal variation in heart rate caused by changes in the rate
    #and depth of breathing.
    "PVC", #Premature Ventricular Contraction (PVC): A ventricular ectopic beat occurring earlier than the next expected sinus beat.
    "PAC", #Premature Atrial Contraction (PAC): An atrial ectopic beat occurring earlier than the next expected sinus beat.
    "AFLT", #Atrial Flutter (AFLT): A supraventricular tachyarrhythmia characterized by organized atrial depolarizations at a rate of 240-320 bpm.
    "SBRAD", #Sinus Bradycardia (SBRAD): A normal rhythm originating from the sinus node with a rate less than 60 bpm.
    "SVTAC", #Supraventricular Tachycardia (SVTAC): A tachyarrhythmia originating above the ventricles.
    "NORM", #Normal (NORM): A normal ECG.
     ]
    
    # Load or preprocess data
    if os.path.exists(X_file) and os.path.exists(y_file):
        X = np.load(X_file)  # Load preprocessed features
        y = np.load(y_file)  # Load preprocessed labels
    else:
        # Load raw ECG signals
        df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        data = []
        total_records = len(df)
        
        for i, f in enumerate(df.filename_lr):
            signal, meta = wfdb.rdsamp(path + f)
            data.append(signal)
            
            # Show progress
            if (i + 1) % 100 == 0 or (i + 1) == total_records:
                print(f"Preprocessing data: {((i + 1) / total_records) * 100:.2f}% complete")
        
        X = np.array(data)  # Extract signals
        
        # Preprocess signals (standardize)
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Process labels into binary format
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)  # Convert label strings to dictionaries
        mlb = MultiLabelBinarizer(classes=classes)
        y = mlb.fit_transform([codes.keys() for codes in df.scp_codes])  # Binarize labels
        
        np.save(X_file, X)  # Save preprocessed features
        np.save(y_file, y)  # Save preprocessed labels
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_model(input_shape=(X.shape[1], X.shape[2]), num_classes=y.shape[1])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define training callbacks
    callbacks = [
        SafeMetricsCallback(validation_data=(X_test, y_test), class_names=classes),
        ReduceLROnPlateau(factor=0.1, patience=5),  # Reduce learning rate on plateau
        EarlyStopping(patience=10, restore_best_weights=True),  # Stop training early if no improvement
        ModelCheckpoint('best_model.keras', save_best_only=True)  # Save the best model
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=classes))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')  # Save the training history plot

if __name__ == "__main__":
    main()
