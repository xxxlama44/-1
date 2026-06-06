import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# =========================================================================
# 1. Define Local Path and Load Data
# =========================================================================
base_path = 'model_ready_data'

print("Loading training and validation data locally...")
try:
    X_train = np.load(os.path.join(base_path, 'X_train.npy'))
    y_train = np.load(os.path.join(base_path, 'y_train.npy'))
    X_val = np.load(os.path.join(base_path, 'X_val.npy'))
    y_val = np.load(os.path.join(base_path, 'y_val.npy'))
    print("All files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: Data files (.npy) not found.\nDetails: {e}")
    exit()

# =========================================================================
# 2. Data Preprocessing & Scaling
# =========================================================================
# Convert labels to One-Hot Encoding format
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)

# Initialize StandardScaler to normalize hand gesture features
scaler = StandardScaler()

# Store original dimensions for accurate reshaping
N_train, T, F = X_train.shape  # N_train=Samples, T=30 (Time steps), F=126 (Features)
N_val, _, _ = X_val.shape

# Reshape to 2D for scaling, then reshape back to 3D (N, 30, 126) for LSTM input
X_train_reshaped = X_train.reshape(-1, F)
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(N_train, T, F)

X_val_reshaped = X_val.reshape(-1, F)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(N_val, T, F)

# Save the scaler locally for real-time camera inference later
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved successfully as 'scaler.pkl'")
print(f"Final Training Data Shape: {X_train_scaled.shape}")
print(f"Final Validation Data Shape: {X_val_scaled.shape}")

# =========================================================================
# 3. Build Lightweight Model Architecture
# =========================================================================
model_final = Sequential([
    # Input shape (30, 126)
    Input(shape=(T, F)), 
    
    # Single Bidirectional LSTM layer for sequence learning
    Bidirectional(LSTM(64, return_sequences=False)),
    BatchNormalization(),
    Dropout(0.5), 
    
    # Dense layer with L2 Regularization to prevent overfitting
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5), 
    
    # 10 output classes for the emergency categories
    Dense(10, activation='softmax') 
])

# Display model architecture summary
model_final.summary()

# =========================================================================
# 4. Compilation, Callbacks & Training
# =========================================================================
# Lower learning rate to ensure stable convergence
opt = Adam(learning_rate=0.0005)
model_final.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Define Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint('Emergency_Model_Final.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("Starting lightweight model training on the local machine...")
history = model_final.fit(
    X_train_scaled, y_train_cat,
    validation_data=(X_val_scaled, y_val_cat),
    epochs=100,         
    batch_size=64,      
    callbacks=callbacks,
    shuffle=True        
)

print("Training complete. The best model version is saved as: Emergency_Model_Final.h5")

# =========================================================================
# 5. Plotting Training Performance (Loss & Accuracy)
# =========================================================================
plt.figure(figsize=(12, 5))

# Loss Subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='#d62728', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#1f77b4', linestyle='--', linewidth=2)
plt.title('Model Loss Reduction', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Accuracy Subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#2ca02c', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Model Accuracy Improvement', fontsize=12)
plt.xlabel('Epochs')
plt.ylabel('Accuracy Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()