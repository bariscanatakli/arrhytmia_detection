import pandas as pd
import numpy as np
import wfdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     MaxPooling1D, Flatten, Dense, Dropout, LSTM, Add,
                                     Attention, GlobalAveragePooling1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

# Load and preprocess the data
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def preprocess_signals(data):
    # Apply bandpass filter to each signal and each channel
    def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100, order=5):
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        try:
            # Apply filtfilt to each channel individually
            filtered = np.array([filtfilt(b, a, channel) if len(channel) > 3 * max(len(a), len(b)) else channel
                                 for channel in data.T]).T
            return filtered
        except ValueError as e:
            print(f"Warning: Data length {len(data)} is too short for filtfilt with padlen=33. Skipping filtering.")
            return data

    filtered_data = np.array([butter_bandpass_filter(signal) for signal in data])
    # Normalize
    scaler = StandardScaler()
    samples, timesteps, channels = filtered_data.shape
    shaped_data = filtered_data.reshape(samples * channels, timesteps)
    normalized_data = scaler.fit_transform(shaped_data)
    return normalized_data.reshape(samples, timesteps, channels)

# Define arrhythmia classes
ARRHYTHMIA_CLASSES = [
    'AFIB',  # Atrial Fibrillation
    'AFLT',  # Atrial Flutter
    'SR',    # Sinus Rhythm
    'STACH', # Sinus Tachycardia
    'SBRAD', # Sinus Bradycardia
    'PVC',   # Premature Ventricular Complex
    'PAC',   # Premature Atrial Complex
    'SVTAC', # Supraventricular Tachycardia
    'SARRH', # Sinus Arrhythmia
    'PACE'   # Pacemaker
]

# Modified aggregate_diagnostic function
def aggregate_diagnostic(y_dic, scp_statements):
    def aggregate(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in scp_statements.index and key in ARRHYTHMIA_CLASSES:
                if scp_statements.loc[key].diagnostic == 1 or scp_statements.loc[key].rhythm == 1:
                    tmp.append(key)
        return list(set(tmp)) if tmp else ['SR']  # Default to SR if no arrhythmia found
    return {key: aggregate(y_dic[key]) for key in y_dic.keys()}

# Build the model with attention mechanism
# Update the create_model function
def create_complex_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # First Conv Block
    x = Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # Second Conv Block
    x = Conv1D(128, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Third Conv Block with residual connection
    x_shortcut = Conv1D(256, kernel_size=1, padding='same')(x)  # Ensure the shapes match
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_shortcut])
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # LSTM layers
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # Attention mechanism
    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)

    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def create_model(input_shape, num_classes):
    model = Sequential([
        # First Conv Block
        Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second Conv Block
        Conv1D(128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layers
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='sigmoid')
    ])
    return model

# Main execution
if __name__ == "__main__":
    # Load data
    path = r'd:/machine-learning-final/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    df = pd.read_csv(path + 'ptbxl_database.csv', index_col=0)
    scp_statements = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    
    # Load and preprocess ECG data
    X = load_raw_data(df, sampling_rate=100, path=path)
    X = preprocess_signals(X)
    
    # Prepare labels
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    y = aggregate_diagnostic(df.scp_codes, scp_statements)
    mlb = MultiLabelBinarizer(classes=ARRHYTHMIA_CLASSES)
    y = mlb.fit_transform(list(y.values()))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Create and compile the complex model
    model = create_complex_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=len(ARRHYTHMIA_CLASSES)
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred_binary,
        target_names=ARRHYTHMIA_CLASSES,
        zero_division=0
    ))